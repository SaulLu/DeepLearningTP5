import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
#from jacobian import JacobianReg
from tqdm import tqdm
from utils import plot3D_traj
import wandb


class Model(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.MSELoss(),
        hidden_size: int = 50,
        lr: float = 1e-3,
        delta_t: float = 1e-2,
        lambda_reg=20,  # lambda jacobian regularisation
        mean=None,
        std=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = criterion
        self.criterion_velocity = nn.L1Loss(reduction='mean')
        self.hidden_size = hidden_size
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.delta_t = delta_t
        self.normalize = True
        self.mean = torch.tensor(mean, dtype=float)
        self.std = torch.tensor(std, dtype=float)

        #self.reg = JacobianReg()

        self.layers = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x):
        if self.normalize or self.mean is None or self.std is None:
            return x + self.layers(x) * self.delta_t
        else:
            x = ((x - self.mean) / self.std).float()
            out = self.layers(x)
            out = x + out * self.delta_t
            out = (out * self.std + self.mean).float()
            return out

    def configure_optimizers(self):
        for param in self.parameters():
            print(f"param: {param.shape}")
        optim_adam = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_adam,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            cooldown=5,
            min_lr=1e-8,
        )
        return (
          [optim_adam],  
          [
            {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'train_loss',
            }
          ]
        )

    def training_step(self, batch, batch_idx):
        data, target, velocity = batch
        #data.requires_grad = True  # this is essential!
        output = self(data)
        velocity_pred = output - data
        loss = self.criterion(output, target) + self.lambda_reg * self.criterion_velocity(velocity_pred, velocity)   # + self.lambda_jr * self.reg(data, output)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, velocity = batch
        output = self(data)
        velocity_pred = output - data
        loss = self.criterion(output, target)  #+ self.lambda_reg * self.criterion_velocity(velocity_pred, velocity) 
        self.log("val_loss", loss, on_epoch=True)
        return {"w_t2": target, "w_t2_pred": output}

    def validation_epoch_end(self, outputs):
        pred_traj = None
        true_traj = None
        for output in outputs:
            w_t2_pred = output["w_t2_pred"].cpu().numpy()
            w_t2 = output["w_t2"].cpu().numpy()

            if pred_traj is None:
                pred_traj = w_t2_pred
                true_traj = w_t2
            true_traj = np.concatenate((true_traj, w_t2), axis=0)
            pred_traj = np.concatenate((pred_traj, w_t2_pred), axis=0)
        ax, fig = plot3D_traj(pred_traj, true_traj)
        self.logger.experiment.log(
            {f"val_traj": wandb.Image(fig), "epoch": self.current_epoch}, commit=True
        )

    def test_step(self, batch, batch_idx):
        data, target, velocity = batch
        output = self(data)
        velocity_pred = output - data
        loss = self.criterion(output, target) #+ self.lambda_reg * self.criterion_velocity(velocity_pred, velocity) 
        self.log("test_loss", loss)

    def full_traj(self, nb_steps, init_pos):
        initial_condition = init_pos[np.newaxis, :]
        print(f"initial_condition: {initial_condition}")

        traj = [torch.tensor(initial_condition, dtype=torch.float)]
        t = np.array([self.delta_t * step for step in range(nb_steps - 1)])
        with torch.no_grad():
            for _ in tqdm(range(nb_steps - 1), position=0, leave=True):
                new_coord = self(traj[-1]).detach()
                traj.append(new_coord)
                # t.append(t[-1] + self.delta_t)
        # traj = np.concatenate(traj, axis=0)
        traj = torch.cat(traj, axis=0)
        traj = traj.numpy()
        print(f"traj: {traj.shape}")
        # t = np.array(t)
        return traj, t

    def jacobian(self, w):
        return torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float))