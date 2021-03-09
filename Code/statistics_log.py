from statistics import (
    lyapunov_exponent,
    newton,
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import logm
import wandb


def compute_pred_true_traj(trained_model, rossler_map_true, initial_condition, nb_step):
    if isinstance(initial_condition, tuple) or isinstance(initial_condition, list):
        initial_condition = np.array(initial_condition)
    traj_pred, time_list = trained_model.full_traj(init_pos=initial_condition, nb_steps=nb_step)
    traj_true, time_list = rossler_map_true.full_traj(init_pos=initial_condition, nb_steps=nb_step)
    return traj_pred, traj_true, time_list


def compute_pred_true_lyaponov(
    wandb_logger,
    trained_model,
    rossler_map_true,
    traj_pred,
    traj_true,
    nb_step,
):
    
    lyap_pred = lyapunov_exponent(
        traj_pred,
        trained_model.jacobian,
        max_it=nb_step,
        delta_t=trained_model.delta_t,
    )

    lyap_true = lyapunov_exponent(
        traj_true,
        rossler_map_true.jacobian,
        max_it=nb_step,
        delta_t=trained_model.delta_t,
    )

    lyap_error = np.abs(lyap_pred - lyap_true)

    print(f"lyap_true: {lyap_true}," f"lyap_pred: {lyap_pred}," f"lyap_error : {lyap_error}")

    if wandb_logger is not None:
        wandb_logger.experiment.log(
            {f"lyap_pred": lyap_pred, "lyap_true": lyap_true, "lyap_error": lyap_error}
        )


def compute_pred_true_equilibrium_state(
    wandb_logger, trained_model, rossler_map_true, mode="discrete"
):
    if mode == "discrete":
        init_pos = np.array([-5.75, -1.6, 0.02])

        # Use the fact that the net is a resnet architecture
        jacobian_system = lambda w: trained_model.jacobian(w).numpy()
        f_system = lambda w: trained_model(torch.tensor(w, dtype=torch.float)).detach().numpy()

        #with torch.no_grad():
        fix_point_pred = newton(f_system, jacobian_system, init_pos)
        fix_point_true = rossler_map_true.equilibrium()
        fix_point_error = np.abs(fix_point_pred - fix_point_true)

        print(f"fix_point_true: {fix_point_true}," f"fix_point_pred: {fix_point_pred}," f"fix_point_error : {fix_point_error}")

        if wandb_logger is not None:
            wandb_logger.experiment.log(
                {f"fix_point_true": fix_point_true, "fix_point_pred": fix_point_pred, "fix_point_error": fix_point_error}
            )
    else:
        raise NotImplementedError()
        
        
        
from statistics import (
    lyapunov_exponent,
    newton,
    plot3D_traj,
    plot_x_pdf,
    plot_x_traj,
    plot_y_pdf,
    plot_y_traj,
    plot_z_pdf,
    plot_z_traj,
)

import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb


def compute_pred_true_traj(trained_model, rossler_map_true, initial_condition, nb_step):
    if isinstance(initial_condition, tuple) or isinstance(initial_condition, list):
        initial_condition = np.array(initial_condition)
    traj_pred, time_list = trained_model.full_traj(init_pos=initial_condition, nb_steps=nb_step)
    traj_true, time_list = rossler_map_true.full_traj(init_pos=initial_condition, nb_steps=nb_step)
    return traj_pred, traj_true, time_list




def compute_pred_true_lyaponov(
    wandb_logger,
    trained_model,
    rossler_map_true,
    traj_pred,
    traj_true,
    nb_step,
    mode="discrete",
):
    if mode in ["discrete", "continuous"]:
        lyap_pred = lyapunov_exponent(
            traj_pred,
            trained_model.jacobian,
            max_it=nb_step,
            delta_t=trained_model.delta_t,
            mode=mode,
        )

        lyap_true = lyapunov_exponent(
            traj_true,
            rossler_map_true.jacobian,
            max_it=nb_step,
            delta_t=trained_model.delta_t,
            mode=mode,
        )

        lyap_error = np.abs(lyap_pred - lyap_true)

        print(f"lyap_true: {lyap_true}," f"lyap_pred: {lyap_pred}," f"lyap_error : {lyap_error}")

        if wandb_logger is not None:
            wandb_logger.experiment.log(
                {f"lyap_pred": lyap_pred, "lyap_true": lyap_true, "lyap_error": lyap_error}
            )
    else:
        raise NotImplementedError()


def compute_pred_true_equilibrium_state(
    wandb_logger, trained_model, rossler_map_true, mode="discrete"
):
    if mode == "discrete":
        init_pos = np.array([-5.75, -1.6, 0.02])

        # Use the fact that the net is a resnet architecture
        jacobian_system = lambda w: (
            (trained_model.jacobian(w) - torch.eye(3)) / trained_model.delta_t
        ).numpy()

        f_system = (
            lambda w: (trained_model(torch.tensor(w, dtype=torch.float)).cpu().numpy() - w)
            / trained_model.delta_t
        )

        with torch.no_grad():
            fix_point_pred = newton(f_system, jacobian_system, init_pos)

        fix_point_true = rossler_map_true.equilibrium()

        fix_point_error = np.abs(fix_point_pred - fix_point_true)

        print(
            f"fix_point_true: {fix_point_true},"
            f"fix_point_pred: {fix_point_pred},"
            f"fix_point_error : {fix_point_error}"
        )
        if wandb_logger is not None:
            wandb_logger.experiment.log(
                {
                    "fix_point_true": fix_point_true,
                    "fix_point_pred": fix_point_pred,
                    "fix_point_error": fix_point_error,
                }
            )
    else:
        raise NotImplementedError()