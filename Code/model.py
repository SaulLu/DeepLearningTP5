import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from utils import plot3D_traj


class DiscreteModel(pl.LightningModule):
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
        if not torch.is_tensor(self.hparams.mean):
            self.hparams.mean = torch.tensor(self.hparams.mean, dtype=torch.float)
        if not torch.is_tensor(self.hparams.std):
            self.hparams.std = torch.tensor(self.hparams.std, dtype=torch.float)

        self.normalize = True

        self.layers = nn.Sequential(
            nn.Linear(3, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, 3),
        )

    def forward(self, x):
        if self.normalize or self.hparams.mean is None or self.hparams.std is None:
            return x + self.layers(x) * self.hparams.delta_t
        else:
            x = ((x - self.hparams.mean) / self.hparams.std).float()
            out = self.layers(x)
            out = x + out * self.hparams.delta_t
            out = (out * self.hparams.std + self.hparams.mean).float()
            return out

    def configure_optimizers(self):
        optim_adam = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optim_adam

    def training_step(self, batch, batch_idx):
        w_t1, w_t2, w_next = batch
        w_t2_pred = self(w_t1)
        w_next_pred = self.full_traj(11, w_t1, return_numpy=False)

        loss_w_t2 = self.hparams.criterion(w_t2, w_t2_pred)
        loss_w_next = self.hparams.criterion_2(w_next, w_next_pred)
        if len(loss_w_next.shape) != 0:
            loss_w_next = loss_w_next.sum() / loss_w_next.shape[0]
        loss = loss_w_t2 + loss_w_next

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        w_t1, w_t2, w_next = batch
        w_t2_pred = self(w_t1)
        w_next_pred = self.full_traj(11, w_t1, return_numpy=False)
        loss_w_t2 = self.hparams.criterion(w_t2, w_t2_pred)
        loss_w_next = self.hparams.criterion_2(w_next, w_next_pred)
        if len(loss_w_next.shape) != 0:
            loss_w_next = loss_w_next.sum() / loss_w_next.shape[0]
        loss = loss_w_t2 + loss_w_next
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse_w_t2", loss_w_t2, on_epoch=True)
        self.log("val_mse_w_next", loss_w_next, on_epoch=True)
        return {"w_next": w_next[0], "w_next_pred": w_next_pred[0]}

    def validation_epoch_end(self, outputs):

        pred_traj = outputs[-1]["w_next_pred"].cpu().numpy()
        true_traj = outputs[-1]["w_next"].cpu().numpy()
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
        w_t1, w_t2, w_next = batch
        w_t2_pred = self(w_t1)
        w_next_pred = self.full_traj(11, w_t1, return_numpy=False)
        loss_w_t2 = self.hparams.criterion(w_t2, w_t2_pred)
        loss_w_next = self.hparams.criterion_2(w_next, w_next_pred)
        if len(loss_w_next.shape) != 0:
            loss_w_next = loss_w_next.sum() / loss_w_next.shape[0]
        loss = loss_w_t2 + loss_w_next
        self.log("test_mse", loss)

    def full_traj(self, nb_steps, init_pos, return_numpy=True):
        if isinstance(init_pos, np.ndarray):
            if len(init_pos.shape) == 1:
                init_pos = init_pos[np.newaxis, :]
            init_pos = torch.tensor(init_pos, dtype=torch.float)

        traj = [init_pos]

        if return_numpy:
            with torch.no_grad():
                for _ in tqdm(range(nb_steps - 1), position=0, leave=True):
                    new_coord = self(traj[-1]).detach()
                    traj.append(new_coord)
        else:
            for _ in range(nb_steps - 1):
                new_coord = self(traj[-1])
                traj.append(new_coord)

        traj = torch.stack(traj, axis=1)
        if return_numpy:
            t = np.array([self.hparams.delta_t * step for step in range(nb_steps - 1)])
            traj = traj.squeeze()
            traj = traj.numpy()
            return traj, t
        return traj

    def jacobian(self, w):
        if torch.is_tensor(w):
            return torch.autograd.functional.jacobian(self, w)
        else:
            return torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float))
