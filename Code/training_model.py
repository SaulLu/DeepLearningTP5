from argparse import ArgumentParser


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from data import DiscreteRosslerAttractorDataModule as RosslerAttractorDataModule
from model import DiscretModel as Model
from rossler_map import RosslerMap
from time_series import Rossler_model
from pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW
from statistics_log import (
    plot_pred_true_trajectories,
    compute_pred_true_traj,
    compute_pred_true_equilibrium_state,
    compute_pred_true_lyaponov,
)


def main(args):
    wandb_logger = WandbLogger(project=args.project)

    checkpoint_callback = ModelCheckpoint(
        # filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="train_mse",
        mode="min",
    )

    datamodule = RosslerAttractorDataModule(
        n_iter_train=args.n_iter_train,
        n_iter_valid=args.n_iter_valid,
        n_iter_test=args.n_iter_test,
        init_pos_train=args.init_pos_train,
        init_pos_test=args.init_pos_train,
        init_pos_valid=args.init_pos_valid,
        batch_size=args.batch_size,
        delta_t=args.delta_t,
    )

    datamodule.setup()

    # tf = args.n_iter_train * args.delta_t
    # time_span = np.linspace(0, tf, args.n_iter_train)
    # time_list = time_span.tolist()

    # sdtw = SoftDTW(use_cuda=False if args.gpus is None else True, gamma=0.1)

    criterion = nn.MSELoss(reduction="mean")

    model = Model(
        criterion=criterion,
        # time_list=time_list,
        # t1=tf,
        lr=args.lr,
        delta_t=args.delta_t,
        mean=datamodule.dataset_train.mean,
        std=datamodule.dataset_train.std,
    )

    trainer = Trainer(
        gpus=args.gpus,
        logger=wandb_logger,
        # auto_lr_find=True,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )

    # trainer.tune(model, datamodule)  # RUn only on CPU mode

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)

    #### Tests ####

    TRAJECTORY_DUR = 10000
    nb_steps = int(TRAJECTORY_DUR // args.delta_t)

    trained_model = Model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    trained_model.normalize = False

    rossler_map_true = RosslerMap(delta_t=args.delta_t)

    # Train set
    traj_pred, traj_true, time_list = compute_pred_true_traj(
        trained_model, rossler_map_true, args.init_pos_train, nb_steps
    )
    plot_pred_true_trajectories(wandb_logger, traj_pred, traj_true, time_list, prefix="train_")
    compute_pred_true_lyaponov(
        wandb_logger,
        trained_model,
        rossler_map_true,
        traj_pred,
        traj_true,
        nb_steps,
        mode="discrete",
    )
    compute_pred_true_equilibrium_state(
        wandb_logger, trained_model, rossler_map_true, mode="discrete"
    )

    # Val set
    traj_pred, traj_true, time_list = compute_pred_true_traj(
        trained_model, rossler_map_true, args.init_pos_valid, nb_steps
    )
    plot_pred_true_trajectories(wandb_logger, traj_pred, traj_true, time_list, prefix="valid_")

    # Test set
    traj_pred, traj_true, time_list = compute_pred_true_traj(
        trained_model, rossler_map_true, args.init_pos_test, nb_steps
    )
    plot_pred_true_trajectories(wandb_logger, traj_pred, traj_true, time_list, prefix="test_")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_iter_train", type=int, default=1000)
    parser.add_argument("--n_iter_valid", type=int, default=1000)
    parser.add_argument("--n_iter_test", type=int, default=1000)
    parser.add_argument("--init_pos_train", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    parser.add_argument("--init_pos_valid", nargs="+", type=float, default=[-5.70, -1.5, -0.02])
    parser.add_argument("--init_pos_test", nargs="+", type=float, default=[0.01, 2.5, 3.07])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--delta_t", type=float, default=1e-2)

    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--gpus", default=None)
    parser.add_argument("--project", type=str, default="test_rossler")

    args = parser.parse_args()
    main(args)