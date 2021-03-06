import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from tqdm import tqdm

from utils import plot3D_traj

TRAJECTORY_DUR = 10000


class DiscreteModel(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.MSELoss(),
        criterion_2=nn.MSELoss(),
        hidden_size: int = 50,
        lr: float = 1e-3,
        delta_t: float = 1e-3,
        mean=None,
        std=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = criterion
        self.hidden_size = hidden_size
        self.lr = lr
        self.delta_t = delta_t
        self.normalize = True
        self.mean = torch.tensor(mean, dtype=float)
        self.std = torch.tensor(std, dtype=float)

        self.criterion_2 = criterion_2

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
        w_t1, w_t2, w_next = batch
        w_t2_pred = self(w_t1)
        w_next_pred = self.full_traj(11, w_t1, return_numpy=False)
        mse_w_t2 = self.criterion(w_t2, w_t2_pred)
        mse_w_next = self.criterion_2(w_next, w_next_pred)
        loss = mse_w_t2 + mse_w_next

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        w_t1, w_t2, w_next = batch
        w_t2_pred = self(w_t1)
        w_next_pred = self.full_traj(11, w_t1, return_numpy=False)
        mse_w_t2 = self.criterion(w_t2, w_t2_pred)
        mse_w_next = self.criterion_2(w_next, w_next_pred)
        loss = mse_w_t2 + mse_w_next
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse_w_t2", mse_w_t2, on_epoch=True)
        self.log("val_mse_w_next", mse_w_next, on_epoch=True)
        return {"w_next": w_next[0], "w_next_pred": w_next_pred[0]}

    def validation_epoch_end(self, outputs):
        import wandb

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
        mse_w_t2 = self.criterion(w_t2, w_t2_pred)
        mse_w_next = self.criterion_2(w_next, w_next_pred)
        loss = mse_w_t2 + mse_w_next
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
            t = np.linspace(0, (nb_steps - 1) * self.delta_t, nb_steps)
            traj = traj.squeeze()
            traj = traj.numpy()
            return traj, t
        return traj

    def jacobian(self, w):
        if torch.is_tensor(w):
            return torch.autograd.functional.jacobian(self, w)
        else:
            return torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float))


class Rossler_model:
    def __init__(self, model_cls, checkpoint_path, args):
        trained_model = DiscreteModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        trained_model.normalize = False
        trained_model.eval()

        self.rosler_nn = trained_model
        self.nb_steps = int(TRAJECTORY_DUR // self.rosler_nn.hparams.delta_t) + 1 + 1
        self.initial_condition = np.array(args.init)

    def full_traj(self):
        traj, t = self.rosler_nn.full_traj(self.nb_steps, self.initial_condition)

        traj = traj[:, 1]

        t_new = np.linspace(0, TRAJECTORY_DUR, int(TRAJECTORY_DUR // 1e-2) + 1)

        traj_function = interp1d(t, traj, kind="linear")
        traj = traj_function(t_new)

        return traj

    def save_traj(self, y):
        np.save("traj.npy", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    args = parser.parse_args()

    ROSSLER = Rossler_model(
        model_cls=DiscreteModel, checkpoint_path=Path("trained_model.ckpt"), args=args
    )

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)
