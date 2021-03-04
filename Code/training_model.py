from argparse import ArgumentParser
from statistics import plot3D_traj
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from rossler_map import RosslerMap
from data import RosslerAttractorDataModule
from model import Model
from time_series import Rossler_model


def main(args):
    wandb_logger = WandbLogger(project=args.project)

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
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

    model = Model(lr=args.lr)

    trainer = Trainer(
        gpus=args.gpus,
        logger=wandb_logger,
        auto_lr_find=True,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )

    # trainer.tune(model, datamodule) # RUn only on CPU mode

    trainer.fit(model, datamodule)

    trainer.test(model=model, datamodule=datamodule)

    checkpoint_path = checkpoint_callback.best_model_path

    rossler_model = Rossler_model(
        delta_t=args.delta_t, model_cls=Model, checkpoint_path=checkpoint_path
    )

    rossler_map_true = RosslerMap(delta_t=args.delta_t)

    plot_pred_true_trajectories(
        wandb_logger=wandb_logger,
        rossler_map_true=rossler_map_true,
        rossler_model=rossler_model,
        initial_condition=args.init_pos_train,
        prefix="train_",
    )
    plot_pred_true_trajectories(
        wandb_logger=wandb_logger,
        rossler_map_true=rossler_map_true,
        rossler_model=rossler_model,
        initial_condition=args.init_pos_valid,
        prefix="valid_",
    )
    plot_pred_true_trajectories(
        wandb_logger=wandb_logger,
        rossler_map_true=rossler_map_true,
        rossler_model=rossler_model,
        initial_condition=args.init_pos_test,
        prefix="test_",
    )


def plot_pred_true_trajectories(
    wandb_logger, rossler_map_true, rossler_model, initial_condition, prefix=None
):
    if isinstance(initial_condition, tuple) or isinstance(initial_condition, list):
        initial_condition = np.array(initial_condition)
    traj = rossler_model.full_traj(initial_condition=initial_condition, y_only=False)
    ax, fig = plot3D_traj(traj)
    wandb_logger.experiment.log({f"{prefix if prefix else ''}predicted_traj": wandb.Image(fig)})

    traj_true, _ = rossler_map_true.full_traj(
        init_pos=initial_condition, nb_steps=rossler_model.nb_steps
    )
    ax, fig = plot3D_traj(traj_true)
    wandb_logger.experiment.log({f"{prefix if prefix else ''}true_traj": wandb.Image(fig)})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_iter_train", type=int, default=2000000)
    parser.add_argument("--n_iter_valid", type=int, default=20000)
    parser.add_argument("--n_iter_test", type=int, default=2000000)
    parser.add_argument("--init_pos_train", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    parser.add_argument("--init_pos_valid", nargs="+", type=float, default=[0.01, 2.5, 3.07])
    parser.add_argument("--init_pos_test", nargs="+", type=float, default=[-5.70, -1.5, -0.02])

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--delta_t", type=float, default=1e-2)

    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--gpus", default=None)
    parser.add_argument("--project", type=str, default="test_rossler")

    args = parser.parse_args()
    main(args)