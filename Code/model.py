import numpy as np
import wandb
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jacobian import JacobianReg
from scipy.integrate import solve_ivp
from scipy.linalg import logm
from tqdm import tqdm
from TorchDiffEqPack import odesolve
from statistics import plot3D_traj


class DiscretModel(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.MSELoss(),
        hidden_size: int = 50,
        in_size: int = 3,
        out_size: int = 3,
        lr: float = 1e-3,
        delta_t: float = 1e-3,
        lambda_jr=0.01,  # lambda jacobian regularisation
        mean=None,
        std=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = criterion
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lr = lr
        self.lambda_jr = lambda_jr
        self.delta_t = delta_t
        self.normalize = True
        self.mean = mean
        self.std = std

        self.reg = JacobianReg()

        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
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
        # print(f"parameters: {len(list(self.parameters()))}")
        return [optim_adam], (
            [
                torch.optim.lr_scheduler.MultiStepLR(
                    optim_adam, milestones=[1, 2], gamma=0.1, last_epoch=-1
                )
            ]
        )

    def training_step(self, batch, batch_idx):
        # for param in self.parameters():
        #     if len(param.data.shape) == 2:
        #         print(f"param: {param.data[0][0]}")
        #         break
        w_t1, w_t2 = batch
        # w_t1.requires_grad = True  # this is essential!

        w_t2_pred = self(w_t1)
        mse = self.criterion(w_t2_pred, w_t2)
        loss = mse  # + self.lambda_jr * self.reg(w_t1, w_t2_pred)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_mse", mse, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        w_t1, w_t2 = batch
        # w_t1.requires_grad = True  # this is essential!

        w_t2_pred = self(w_t1)
        mse = self.criterion(w_t2_pred, w_t2)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse", mse, on_epoch=True)
        return {"w_t2": w_t2, "w_t2_pred": w_t2_pred}

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
        print(f"true_traj: {true_traj.shape}")
        print(f"pred_traj: {pred_traj.shape}")
        ax, fig = plot3D_traj(pred_traj, true_traj)
        ax.scatter(true_traj[-1][0], true_traj[-1][1], true_traj[-1][2], marker="o", label="true")
        ax.scatter(
            pred_traj[-1][0],
            pred_traj[-1][1],
            pred_traj[-1][2],
            marker="^",
            color="r",
            label="pred",
        )
        ax.legend()
        self.logger.experiment.log(
            {f"val_traj": wandb.Image(fig), "epoch": self.current_epoch}, commit=False
        )

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        mse = self.criterion(output, target)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("test_loss", loss)
        self.log("test_mse", mse)

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
        traj = torch.stack(traj, axis=0)
        traj = traj.numpy()
        print(f"traj: {traj.shape}")
        # t = np.array(t)

        return traj, t

    def jacobian(self):
        """Jacobian of the net

        Returns:
            [type]: [description]
        """
        return lambda w: torch.autograd.functional.jacobian(
            self, torch.tensor(w, dtype=torch.float)
        )
        # return lambda w: (
        #     logm(
        #         (
        #             (
        #                 torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float))
        #                 - torch.eye(3)
        #             )
        #             / delta_t
        #         )
        #         .cpu()
        #         .numpy()
        #     )
        # )
        # return lambda w: (
        #     (torch.autograd.functional.jacobian(self, torch.tensor(w, dtype=torch.float)) - torch.eye(3)) / delta_t
        # )


class ContinuousModel(pl.LightningModule):
    def __init__(
        self,
        time_list: list,
        t1: float,
        criterion=nn.MSELoss(),
        hidden_size: int = 10,
        in_size: int = 3,
        out_size: int = 3,
        lr: float = 1e-3,
        delta_t: float = 1e-2,
        lambda_jr=0.01,  # lambda jacobian regularisation
    ):
        super().__init__()
        self.save_hyperparameters()
        self.time_list = time_list
        self.t1 = t1
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lr = lr
        self.lambda_jr = lambda_jr
        self.delta_t = delta_t

        self.criterion = criterion
        self.reg = JacobianReg()

        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.ReLU(),
        )

        self.ode_block = ODEBlock(self.forward, self.time_list, self.t1)

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

        # time_list = time_list.tolist()

        output = self.ode_block(positions[:, 0, :])
        output = output.permute(1, 0, -1)

        # output = output.view(1, -1)
        # positions = positions.view(1, -1)

        # print(f"output: {output.shape}")
        # print(f"positions: {positions.shape}")

        # print(f"time: {time_list == self.time_list}")
        # print(f"self.time: {time_list[0][10]}")
        # print(f"self.time: {time_list[0][10]}")

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

        # time_list = time_list.tolist()

        output = self.ode_block(positions[:, 0, :])
        # output = output.unsqueeze(0)
        # print(f"output: {output.shape}")

        output = output.permute(1, 0, -1)

        ouputs = {"pred_traj": output[0].cpu(), "true_traj": positions[0].cpu()}

        output = torch.flatten(output, start_dim=1)
        positions = torch.flatten(positions, start_dim=1)
        # print(f"output: {output.shape}")
        # print(f"positions: {positions.shape}")

        mse = self.criterion(output, positions)
        # loss = mse + self.lambda_jr * self.reg(data, output)

        # self.log("val_loss", loss, on_epoch=True)
        self.log("val_mse", mse, on_epoch=True)
        return ouputs

    def validation_epoch_end(self, outputs):
        pred_traj = outputs[-1]["pred_traj"]
        true_traj = outputs[-1]["true_traj"]
        ax, fig = plot3D_traj(pred_traj, true_traj)
        self.logger.experiment.log(
            {f"val_traj": wandb.Image(fig), "epoch": self.current_epoch}, commit=False
        )

    def test_step(self, batch, batch_idx):
        positions, time_list = batch
        positions.requires_grad = True  # this is essential!

        # print(f"positions: {positions.shape}")
        # print(f"time_list: {time_list.shape}")
        # print(f"time_list: {time_list[-1]}")

        # time_list = time_list.tolist()

        output = self.ode_block(positions[:, 0, :])
        output = output.permute(1, 0, -1)

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
    def __init__(self, odefunc, time_list, t1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        options = {}
        options.update({"method": "Dopri5"})
        options.update({"h": None})
        options.update({"t0": 0.0})
        options.update({"t1": t1})
        options.update({"rtol": 1e-7})
        options.update({"atol": 1e-8})
        options.update({"print_neval": False})
        options.update({"neval_max": 1000000})
        options.update({"t_eval": time_list})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})
        self.options = options

    def forward(self, init_pos):
        # tf = time_list[-1]
        # print(f"tf: {tf}")
        # options = {}
        # options.update({"method": "Dopri5"})
        # options.update({"h": None})
        # options.update({"t0": 0.0})
        # options.update({"t1": tf})
        # options.update({"rtol": 1e-7})
        # options.update({"atol": 1e-8})
        # options.update({"print_neval": False})
        # options.update({"neval_max": 1000000})
        # options.update({"t_eval": time_list})
        # options.update({"interpolation_method": "cubic"})
        # options.update({"regenerate_graph": False})
        out = odesolve(self.odefunc, init_pos, self.options)
        # out = out.permute(1, 0, -1)
        return out
