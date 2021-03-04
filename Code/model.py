import pytorch_lightning as pl
import torch
import torch.nn as nn
from jacobian import JacobianReg
import numpy as np


class Model(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 10,
        in_size: int = 3,
        out_size: int = 3,
        lr: float = 1e-3,
        delta_t: float = 1e-3,
        lambda_jr=0.01,  # lambda jacobian regularisation
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lr = lr
        self.lambda_jr = lambda_jr
        self.delta_t = delta_t

        self.criterion = nn.MSELoss()
        self.reg = JacobianReg()

        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x) * self.delta_t

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data.requires_grad = True  # this is essential!

        output = self(data)
        mse = self.criterion(output, target)
        loss = mse + self.lambda_jr * self.reg(data, output)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        mse = self.criterion(output, target)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse", mse, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        mse = self.criterion(output, target)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("test_loss", loss)
        self.log("test_mse", mse)

    def full_traj(self, trajectory_duration, initial_condition):
        nb_steps = int(trajectory_duration // self.delta_t)

        traj = [initial_condition[np.newaxis, :]]
        t = [self.delta_t]
        with torch.no_grad():
            for _ in range(nb_steps - 1):
                new_coord = self(torch.tensor(traj[-1], dtype=torch.float)).numpy()
                traj.append(new_coord)
                t.append(t[-1] + self.delta_t)
        traj = np.concatenate(traj, axis=0)
        t = np.array(t)

        return traj, t