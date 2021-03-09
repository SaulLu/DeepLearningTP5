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
        # for param in self.parameters():
        #     print(f"param: {param.shape}")
        optim_adam = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim_adam

    def training_step(self, batch, batch_idx):
        w_t1, w_t2, w_next = batch
        # print(f"w_t2: {w_t2.shape}")
        # print(f"w_next: {w_next.shape}")
        # data.requires_grad = True  # this is essential!
        w_t2_pred = self(w_t2)
        w_next_pred = self.full_traj(10, w_t1, return_numpy=False)
        mse_w_t2 = self.criterion(w_t2, w_t2_pred)  # + self.lambda_jr * self.reg(data, output)
        mse_w_next = self.criterion(w_next, w_next_pred)
        loss = mse_w_t2 + mse_w_next

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # self.log("train_mse_w_t2", mse_w_t2, on_epoch=True)
        # self.log("train_mse_w_next", mse_w_next, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        w_t1, w_t2, w_next = batch
        # print(f"w_t2: {w_t2.shape}")
        # print(f"w_next: {w_next.shape}")
        # data.requires_grad = True  # this is essential!
        w_t2_pred = self(w_t2)
        w_next_pred = self.full_traj(10, w_t1, return_numpy=False)
        mse_w_t2 = self.criterion(w_t2, w_t2_pred)  # + self.lambda_jr * self.reg(data, output)
        mse_w_next = self.criterion(w_next, w_next_pred)
        loss = mse_w_t2 + mse_w_next
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse_w_t2", mse_w_t2, on_epoch=True)
        self.log("val_mse_w_next", mse_w_next, on_epoch=True)
        return {"w_next": w_next[0], "w_next_pred": w_next_pred[0]}

    def validation_epoch_end(self, outputs):
        pred_traj = outputs[-1]["w_next_pred"].cpu().numpy()
        true_traj = outputs[-1]["w_next"].cpu().numpy()
        # pred_traj = None
        # true_traj = None
        # for output in outputs:
        #     w_t2_pred = output["w_t2_pred"].cpu().numpy()
        #     w_t2 = output["w_t2"].cpu().numpy()

        #     if pred_traj is None:
        #         pred_traj = w_t2_pred
        #         true_traj = w_t2
        #     true_traj = np.concatenate((true_traj, w_t2), axis=0)
        #     pred_traj = np.concatenate((pred_traj, w_t2_pred), axis=0)
        # print(f"true_traj: {true_traj.shape}")
        # print(f"pred_traj: {pred_traj.shape}")
        ax, fig = plot3D_traj(pred_traj, true_traj)
        # ax.scatter(true_traj[-1][0], true_traj[-1][1], true_traj[-1][2], marker="o", label="true")
        # ax.scatter(
        #     pred_traj[-1][0],
        #     pred_traj[-1][1],
        #     pred_traj[-1][2],
        #     marker="^",
        #     color="r",
        #     label="pred",
        # )
        ax.legend()
        self.logger.experiment.log(
            {"val_traj": wandb.Image(fig), "epoch": self.current_epoch}, commit=False
        )

    def test_step(self, batch, batch_idx):
        w_t1, w_t2, w_next = batch
        # print(f"w_t2: {w_t2.shape}")
        # print(f"w_next: {w_next.shape}")
        # data.requires_grad = True  # this is essential!
        w_t2_pred = self(w_t2)
        w_next_pred = self.full_traj(10, w_t1, return_numpy=False)
        mse_w_t2 = self.criterion(w_t2, w_t2_pred)  # + self.lambda_jr * self.reg(data, output)
        mse_w_next = self.criterion(w_next, w_next_pred)
        loss = mse_w_t2 + mse_w_next

        # self.log("test_loss", loss)
        self.log("test_mse", loss)

    def full_traj(self, nb_steps, init_pos, return_numpy=True):
        if isinstance(init_pos, np.ndarray):
            if len(init_pos.shape) == 1:
                init_pos = init_pos[np.newaxis, :]
            init_pos = torch.tensor(init_pos, dtype=torch.float)
        # print(f"initial_condition: {init_pos.shape}")

        traj = [init_pos]

        with torch.no_grad():
            if return_numpy:
                for _ in tqdm(range(nb_steps - 1), position=0, leave=True):
                    new_coord = self(traj[-1]).detach()
                    # print(f"new_coord: {new_coord.shape}")

                    traj.append(new_coord)
            else:
                for _ in range(nb_steps - 1):
                    new_coord = self(traj[-1])
                    # print(f"new_coord: {new_coord.shape}")

                    traj.append(new_coord)
                # t.append(t[-1] + self.delta_t)
        # traj = np.concatenate(traj, axis=0)
        # traj = torch.cat(traj, axis=1)
        traj = torch.stack(traj, axis=1)
        if return_numpy:
            t = np.array([self.delta_t * step for step in range(nb_steps - 1)])
            traj = traj.squeeze()
            traj = traj.numpy()
            # print(f"traj: {traj.shape}")
            return traj, t
        # print(f"traj: {traj.shape}")
        # t = np.array(t)

        return traj

    def jacobian(self, w):
        return torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float))
