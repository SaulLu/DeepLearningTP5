import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.linalg import norm, qr, solve
from scipy.fftpack import fft, fftshift
from scipy.linalg import expm

import wandb


def plot3D_traj(traj_pred, traj_true):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], "-.", label="true")
    ax.plot(traj_pred[:, 0], traj_pred[:, 1], traj_pred[:, 2], "-.", label="pred")
    ax.legend()
    return ax, fig


def compute_traj(trained_model, rossler_map_true, initial_condition, nb_step):
    if isinstance(initial_condition, tuple) or isinstance(initial_condition, list):
        initial_condition = np.array(initial_condition)
    traj_pred, time_list = trained_model.full_traj(init_pos=initial_condition, nb_steps=nb_step)
    traj_true, time_list = rossler_map_true.full_traj(init_pos=initial_condition, nb_steps=nb_step)
    return traj_pred, traj_true, time_list


class Statistics:
    def __init__(self, wandb_logger, ts_n=10000, fft_n=1000):
        self.wandb_logger = wandb_logger
        self.fft_n = fft_n
        self.ts_n = ts_n
        self.axis_names = ["x", "y", "z"]

    def add_traj(self, traj_true, traj_pred, time_list, prefix=None):
        self.traj_true = traj_true
        self.traj_pred = traj_pred
        self.time_list = time_list
        self.prefix = prefix

    def log_plot(self, ax, fig, title):
        prefix = self.prefix
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.log(
                {f"{prefix if prefix else ''}{title}": wandb.Image(fig)}
            )
        else:
            plt.title(title)
            plt.show()

    def plot_pdf(self):
        fig, ax = plt.subplots(1, 3)
        fig.set_figwidth(15)
        fig.set_figheight(5)
        for idx in range(3):
            ax[idx].hist(self.traj_true[:, idx], label="true", alpha=0.8)
            ax[idx].hist(self.traj_pred[:, idx], label="pred", alpha=0.8)
            ax[idx].legend()
            ax[idx].set_ylabel(self.axis_names[idx])
        self.log_plot(ax, fig, "PDF")

    def plot1D_traj(self):
        T = self.ts_n
        fig, ax = plt.subplots(3, 1)
        fig.set_figwidth(15)
        fig.set_figheight(15)
        for idx in range(3):
            ax[idx].plot(self.time_list[:T], self.traj_true[:T, idx], "-.", label="true")
            ax[idx].plot(self.time_list[:T], self.traj_pred[:T, idx], "-.", label="pred")
            ax[idx].legend()
            ax[idx].set_ylabel(self.axis_names[idx])
        self.log_plot(ax, fig, "1D Trajectories")

    def plot_corr(self):
        T = self.ts_n
        fig, ax = plt.subplots(3, 1)
        fig.set_figwidth(15)
        fig.set_figheight(15)
        for idx in range(3):
            corr_true = np.correlate(self.traj_true[:T, idx], self.traj_true[:T, idx], "same")
            corr_pred = np.correlate(self.traj_pred[:T, idx], self.traj_pred[:T, idx], "same")
            ax[idx].plot(self.time_list[:T], corr_true, "-.", label="true")
            ax[idx].plot(self.time_list[:T], corr_pred, "-.", label="pred")
            ax[idx].legend()
            ax[idx].set_ylabel(self.axis_names[idx])
        self.log_plot(ax, fig, "Time correlations")

    def plot_fft(self):
        fig, ax = plt.subplots(3, 1, figsize=(15, 15))
        for idx in range(3):
            spec_true = self.compute_spec(self.traj_true[:, idx])
            spec_pred = self.compute_spec(self.traj_pred[:, idx])
            ax[idx].plot(spec_true, "-.", label="true")
            ax[idx].plot(spec_pred, "-.", label="pred")
            ax[idx].legend()
            ax[idx].set_ylabel(self.axis_names[idx])
        self.log_plot(ax, fig, "FFT")

    def compute_spec(self, w):
        spec = fft(w, self.fft_n, axis=0)
        spec = np.abs(fftshift(spec, axes=0))
        spec /= np.max(spec)
        spec_log = 20 * np.log10(spec + np.finfo(np.float32).eps)
        return spec_log

    def plot3D_traj(self):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot(
            self.traj_true[:, 0],
            self.traj_true[:, 1],
            self.traj_true[:, 2],
            "-.",
            label="true",
        )
        ax.plot(
            self.traj_pred[:, 0],
            self.traj_pred[:, 1],
            self.traj_pred[:, 2],
            "-.",
            label="pred",
        )
        ax.legend()
        self.log_plot(ax, fig, "3D Trajectory")

    def plot_all(self):
        self.plot3D_traj()
        self.plot1D_traj()
        self.plot_pdf()
        self.plot_corr()
        self.plot_fft()


class Dynamics:
    def __init__(self, wandb_logger, true_model, trained_model, max_it=1000, max_comp=100000):
        self.wandb_logger = wandb_logger
        self.true_model = true_model
        self.trained_model = trained_model
        self.delta_t = trained_model.delta_t
        self.max_it = max_it
        self.max_comp = max_comp

    def add_traj(self, traj_true, traj_pred):
        self.traj_true = traj_true
        self.traj_pred = traj_pred

    def lyapunov_exponent(self, traj, jacobian, mode="discrete"):
        print(f"mode: {mode}")
        print(f"traj: {traj.shape}")
        print(f"max_it: {self.max_it}")
        print(f"delta_t: {self.delta_t}")
        print(f"jacobian: {jacobian}")
        n = traj.shape[1]
        w = np.eye(n)
        rs = []
        for i in range(self.max_it):
            jacob = jacobian(traj[i, :])
            if mode == "discrete":
                w_next = np.dot(jacob, w)
            elif mode == "continuous":
                w_next = np.dot(expm(jacob * self.delta_t), w)
        w_next, r_next = qr(w_next)
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())
        rs.append(r_next)
        w = w_next
        return np.mean(np.log(rs), axis=0) / self.delta_t

    def newton(self, f, jacob, x):
        tol = x
        for compt in range(self.compt_max):
            x = x - solve(jacob(x), f(x))
            tol = norm(tol - x)
            if tol <= 1e-5:
                return x
        return x

    def compute_lyaponov(self):

        print("Pred")
        lyap_pred = self.lyapunov_exponent(
            self.traj_pred, self.trained_model.jacobian, mode="discrete"
        )

        print("True")
        lyap_true = self.lyapunov_exponent(
            self.traj_true, self.true_model.jacobian, mode="continuous"
        )
        lyap_error = np.abs(lyap_pred - lyap_true)
        print(f"lyap_true: {lyap_true}," f"lyap_pred: {lyap_pred}," f"lyap_error : {lyap_error}")
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.log(
                {"lyap_pred": lyap_pred, "lyap_true": lyap_true, "lyap_error": lyap_error}
            )

    def compute_equilibrum(self):
        init_pos = np.array([-5.75, -1.6, 0.02])

        # Use the fact that the net is a resnet architecture
        jacobian_system = (
            lambda w: (
                (self.trained_model.jacobian(torch.tensor(w, dtype=torch.float)) - torch.eye(3))
                / self.delta_t
            )
            .cpu()
            .numpy()
        )

        f_system = (
            lambda w: (self.trained_model(torch.tensor(w, dtype=torch.float)).cpu().numpy() - w)
            / self.delta_t
        )

        fix_point_pred = self.newton(f_system, jacobian_system, init_pos)
        fix_point_true = self.true_model.equilibrium()
        fix_point_error = np.abs(fix_point_pred - fix_point_true)

        print(
            f"fix_point_true: {fix_point_true},"
            f"fix_point_pred: {fix_point_pred},"
            f"fix_point_error : {fix_point_error}"
        )

        if self.wandb_logger is not None:
            self.wandb_logger.experiment.log(
                {
                    "fix_point_true": fix_point_true,
                    "fix_point_pred": fix_point_pred,
                    "fix_point_error": fix_point_error,
                }
            )

    def plot_all(self):
        self.compute_lyaponov()
        self.compute_equilibrum()
