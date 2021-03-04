import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jacobian import JacobianReg
from scipy.integrate import solve_ivp

from TorchDiffEqPack import odesolve


class DiscretModel(pl.LightningModule):
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


class ContinuousModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 10,
        in_size: int = 3,
        out_size: int = 3,
        lr: float = 1e-3,
        delta_t: float = 1e-2,
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

        self.ode_block = ODEBlock(self.forward)

    def forward(self, t, x):
        # t = torch.tensor(t, dtype=torch.float)
        # print(f"t: {t}, x:{x}")
        input = x
        return self.layers(input)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        positions, time_list = batch
        # positions.requires_grad = True  # this is essential!

        # print(f"positions: {positions.shape}")
        # print(f"time_list: {time_list.shape}")
        # print(f"time_list: {time_list[-1]}")

        time_list = time_list.tolist()

        output = self.ode_block(positions[0, 0, :], time_list[0])
        output = output.unsqueeze(0)

        # output = output.view(1, -1)
        # positions = positions.view(1, -1)

        # print(f"output: {output.shape}")
        # print(f"positions: {positions.shape}")

        mse = self.criterion(output, positions)
        loss = mse  # + self.lambda_jr * self.reg(positions, output)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        positions, time_list = batch
        # positions.requires_grad = True  # this is essential!

        # print(f"positions: {positions.shape}")
        # print(f"time_list: {time_list.shape}")
        # print(f"time_list: {time_list[-1]}")

        time_list = time_list.tolist()

        output = self.ode_block(positions[0, 0, :], time_list[0])
        output = output.unsqueeze(0)

        # print(f"output: {output.shape}")
        # print(f"positions: {positions.shape}")

        mse = self.criterion(output, positions)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse", mse, on_epoch=True)

    def test_step(self, batch, batch_idx):
        positions, time_list = batch
        positions.requires_grad = True  # this is essential!

        # print(f"positions: {positions.shape}")
        # print(f"time_list: {time_list.shape}")
        # print(f"time_list: {time_list[-1]}")

        time_list = time_list.tolist()

        output = self.ode_block(positions[0, 0, :], time_list[0])
        output = output.unsqueeze(0)

        # print(f"output: {output.shape}")
        # print(f"positions: {positions.shape}")

        mse = self.criterion(output, positions)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("test_loss", loss)
        self.log("test_mse", mse)

    def full_traj(self, trajectory_duration, initial_condition):
        initial_condition = torch.tensor(initial_condition, dtype=torch.float)
        time_span = np.linspace(0, trajectory_duration, int(trajectory_duration // self.delta_t))
        t_list = time_span.tolist()
        options = {}
        options.update({"method": "Dopri5"})
        options.update({"h": None})
        options.update({"t0": 0.0})
        options.update({"t1": trajectory_duration})
        options.update({"rtol": 1e-7})
        options.update({"atol": 1e-8})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": t_list})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        with torch.no_grad():
            traj = odesolve(self, initial_condition, options)
        # print(f"traj: {traj.shape}")
        traj = traj.numpy()
        return traj, time_span


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # options = {}
        # options.update({"method": "Dopri5"})
        # options.update({"h": None})
        # options.update({"t0": 0.0})
        # options.update({"t1": 1.0})
        # options.update({"rtol": 1e-7})
        # options.update({"atol": 1e-8})
        # options.update({"print_neval": False})
        # options.update({"neval_max": 1000000})
        # options.update({"t_eval": None})
        # options.update({"interpolation_method": "cubic"})
        # options.update({"regenerate_graph": False})
        # self.options = options

    def forward(self, init_pos, time_list):
        tf = time_list[-1]
        # print(f"tf: {tf}")
        options = {}
        options.update({"method": "Dopri5"})
        options.update({"h": None})
        options.update({"t0": 0.0})
        options.update({"t1": tf})
        options.update({"rtol": 1e-7})
        options.update({"atol": 1e-8})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": time_list})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        out = odesolve(self.odefunc, init_pos, options)
        # out = out.permute(1, 0, -1)
        return out
