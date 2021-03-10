import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jacobian import JacobianReg
from tqdm import tqdm

import wandb
from utils import plot3D_traj


class Model(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.MSELoss(),
        criterion_2=nn.MSELoss(),
        hidden_size: int = 50,
        lr: float = 1e-3,
        delta_t: float = 1e-3,
        lambda_jr=0.01,  # lambda jacobian regularisation
        mean=None,
        std=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = criterion
        self.hidden_size = hidden_size
        self.lr = lr
        self.lambda_jr = lambda_jr
        self.delta_t = delta_t
        self.normalize = True
        self.mean = torch.tensor(mean, dtype=float)
        self.std = torch.tensor(std, dtype=float)

        self.criterion_2 = criterion_2

        self.reg = JacobianReg()

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
        optim_adam = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim_adam

    def training_step(self, batch, batch_idx):
        w_t1, w_t2, _ = batch
        w_t2_pred = self(w_t1)

        loss = self.criterion(w_t2, w_t2_pred)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        w_t1, w_t2, w_10 = batch
        w_t2_pred = self(w_t1)

        loss = self.criterion(w_t2, w_t2_pred)

        self.log("val_loss", loss, on_epoch=True)
        return {"w_t1": w_t1, "w_10": w_10}

    def validation_epoch_end(self, outputs):
        w_t1 = outputs[-1]["w_t1"][0]
        pred_traj, t = self.full_traj(11, w_t1, return_numpy=True)
        # print(f"pred_traj: {pred_traj.shape}")

        # pred_traj = outputs[-1]["w_next_pred"].cpu().numpy()
        true_traj = outputs[-1]["w_10"].cpu().numpy()[0]
        # print(f"true_traj: {true_traj.shape}")

        ax, fig = plot3D_traj(pred_traj, true_traj)
        ax.scatter(true_traj[0][0], true_traj[0][1], true_traj[0][2], marker="o", label="true")
        ax.scatter(
            pred_traj[0][0],
            pred_traj[0][1],
            pred_traj[0][2],
            marker="^",
            color="r",
            label="pred",
        )
        ax.legend()
        self.logger.experiment.log(
            {"val_traj": wandb.Image(fig), "epoch": self.current_epoch}, commit=False
        )

    def test_step(self, batch, batch_idx):
        w_t1, w_t2, _ = batch
        w_t2_pred = self(w_t1)

        loss = self.criterion(w_t2, w_t2_pred)

        self.log("test_mse", loss)

    def full_traj(self, nb_steps, init_pos, return_numpy=True):
        if isinstance(init_pos, np.ndarray):
            init_pos = torch.tensor(init_pos, dtype=torch.float)
        if len(init_pos.shape) == 1:
            init_pos = init_pos.unsqueeze(0)

        traj = [init_pos]

        with torch.no_grad():
            if return_numpy:
                for _ in tqdm(range(nb_steps - 1), position=0, leave=True):
                    new_coord = self(traj[-1]).detach()
                    traj.append(new_coord)
            else:
                for _ in range(nb_steps - 1):
                    new_coord = self(traj[-1])
                    traj.append(new_coord)
        traj = torch.stack(traj, axis=1)
        if return_numpy:
            t = np.array([self.delta_t * step for step in range(nb_steps - 1)])
            traj = traj.squeeze()
            traj = traj.cpu().numpy()
            return traj, t

        return traj

    def jacobian(self, w):
        if torch.is_tensor(w):
            return torch.autograd.functional.jacobian(self, w)
        return torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float))
