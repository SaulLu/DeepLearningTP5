import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

# import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import RosslerAttractorDataModule
from model import DiscreteModel
from pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW
from rossler_map import RosslerMap
from utils import Dynamics, Statistics, compute_traj


def main(args):
    # wandb_logger = WandbLogger(project=args.project)

    # checkpoint_callback = ModelCheckpoint(
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val_loss",
    #     mode="min",
    # )

    # datamodule = RosslerAttractorDataModule(
    #     n_iter_train=args.n_iter_train,
    #     n_iter_valid=args.n_iter_valid,
    #     n_iter_test=args.n_iter_test,
    #     init_pos_train=args.init_pos_train,
    #     init_pos_test=args.init_pos_train,
    #     init_pos_valid=args.init_pos_valid,
    #     batch_size=args.batch_size,
    #     delta_t=args.delta_t,
    # )

    # datamodule.setup()

    # criterion = nn.L1Loss(reduction="mean")
    # use_cuda = False if args.gpus is None else True
    # criterion_2 = SoftDTW(use_cuda=use_cuda, gamma=0.1, normalize=True)
    # # criterion_2 = nn.MSELoss(reduction="mean")

    # # checkpoint_path = "Data/checkpoints/model_dtw.ckpt"

    # # model = DiscreteModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # # model.hparams.criterion_2 = criterion_2
    # # model.configure_optimizers()
    # # model.hparams.lr = args.lr

    # model = DiscreteModel(
    #     criterion=criterion,
    #     criterion_2=criterion_2,
    #     lr=args.lr,
    #     delta_t=args.delta_t,
    #     mean=datamodule.dataset_train.mean,
    #     std=datamodule.dataset_train.std,
    #     hidden_size=15,
    # )

    # trainer = Trainer(
    #     gpus=args.gpus,
    #     logger=wandb_logger,
    #     max_epochs=args.epochs,
    #     callbacks=[checkpoint_callback],
    #     # auto_lr_find=True,
    # )

    # # trainer.tune(model=model, datamodule=datamodule)

    # trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model=model, datamodule=datamodule)

    # # Tests

    # TRAJECTORY_DUR = 1000
    # nb_steps = int(TRAJECTORY_DUR // args.delta_t)

    # checkpoint_path = Path(checkpoint_callback.best_model_path)
    # save_dir_path = checkpoint_path.parent

    # trained_model = DiscreteModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    # trained_model.normalize = False

    # true_model = RosslerMap(delta_t=args.delta_t)

    # statstics_calculator = Statistics(wandb_logger)
    # dynamics_calculator = Dynamics(wandb_logger, true_model, trained_model, nb_steps)

    # # TRAIN set
    # traj_pred, traj_true, time_list = compute_traj(
    #     trained_model, true_model, args.init_pos_train, nb_steps
    # )
    # np.save(os.path.join(save_dir_path, "traj_pred_train.npy"), traj_pred)
    # np.save(os.path.join(save_dir_path, "traj_true_train.npy"), traj_true)
    # np.save(os.path.join(save_dir_path, "time_list_train.npy"), time_list)

    # statstics_calculator.add_traj(traj_true, traj_pred, time_list, prefix="train ")
    # statstics_calculator.plot_all()
    # dynamics_calculator.add_traj(traj_true, traj_pred)
    # dynamics_calculator.plot_all()

    # # VAL set
    # # traj_pred, traj_true, time_list = compute_traj(
    # #     trained_model, true_model, args.init_pos_valid, nb_steps
    # # )
    # # np.save(os.path.join(save_dir_path, "traj_pred_valid.npy"), traj_pred)
    # # np.save(os.path.join(save_dir_path, "traj_true_valid.npy"), traj_true)
    # # np.save(os.path.join(save_dir_path, "time_list_valid.npy"), time_list)

    # # statstics_calculator.add_traj(traj_true, traj_pred, time_list, prefix="valid ")
    # # statstics_calculator.plot_all()
    # # dynamics_calculator.add_traj(traj_true, traj_pred)
    # # dynamics_calculator.plot_all()

    # # TEST set
    # traj_pred, traj_true, time_list = compute_traj(
    #     trained_model, true_model, args.init_pos_test, nb_steps
    # )
    # np.save(os.path.join(save_dir_path, "traj_pred_test.npy"), traj_pred)
    # np.save(os.path.join(save_dir_path, "traj_true_test.npy"), traj_true)
    # np.save(os.path.join(save_dir_path, "time_list_test.npy"), time_list)

    # statstics_calculator.add_traj(traj_true, traj_pred, time_list, prefix="test ")
    # statstics_calculator.plot_all()
    # dynamics_calculator.add_traj(traj_true, traj_pred)
    # dynamics_calculator.plot_all()

    # # Training 2

    wandb_logger = WandbLogger(project=args.project)

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

    # criterion = nn.L1Loss(reduction="mean")
    use_cuda = False if args.gpus is None else True
    criterion_2 = SoftDTW(use_cuda=use_cuda, gamma=0.1, normalize=True)
    # criterion_2 = nn.MSELoss(reduction="mean")

    checkpoint_path = Path(
        "/content/drive/My Drive/DeepLearningTP5/Code/checkpoints/trained_model.ckpt"
    )

    model = DiscreteModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.hparams.criterion_2 = criterion_2
    # model.hparams.lr = args.lr
    # model.configure_optimizers()

    # model = DiscreteModel(
    #     criterion=criterion,
    #     criterion_2=criterion_2,
    #     lr=args.lr,
    #     delta_t=args.delta_t,
    #     mean=datamodule.dataset_train.mean,
    #     std=datamodule.dataset_train.std,
    #     hidden_size=15,
    # )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        gpus=args.gpus,
        logger=wandb_logger,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        # auto_lr_find=True,
    )

    # trainer.tune(model=model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    # Tests

    TRAJECTORY_DUR = 1000
    nb_steps = int(TRAJECTORY_DUR // args.delta_t)

    checkpoint_path = Path(checkpoint_callback.best_model_path)
    save_dir_path = checkpoint_path.parent

    trained_model = DiscreteModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    trained_model.normalize = False

    true_model = RosslerMap(delta_t=args.delta_t)

    statstics_calculator = Statistics(wandb_logger)
    dynamics_calculator = Dynamics(wandb_logger, true_model, trained_model, nb_steps)

    # TRAIN set
    traj_pred, traj_true, time_list = compute_traj(
        trained_model, true_model, args.init_pos_train, nb_steps
    )
    np.save(os.path.join(save_dir_path, "traj_pred_train.npy"), traj_pred)
    np.save(os.path.join(save_dir_path, "traj_true_train.npy"), traj_true)
    np.save(os.path.join(save_dir_path, "time_list_train.npy"), time_list)

    statstics_calculator.add_traj(traj_true, traj_pred, time_list, prefix="train ")
    statstics_calculator.plot_all()
    dynamics_calculator.add_traj(traj_true, traj_pred)
    dynamics_calculator.plot_all()

    # VAL set
    # traj_pred, traj_true, time_list = compute_traj(
    #     trained_model, true_model, args.init_pos_valid, nb_steps
    # )
    # np.save(os.path.join(save_dir_path, "traj_pred_valid.npy"), traj_pred)
    # np.save(os.path.join(save_dir_path, "traj_true_valid.npy"), traj_true)
    # np.save(os.path.join(save_dir_path, "time_list_valid.npy"), time_list)

    # statstics_calculator.add_traj(traj_true, traj_pred, time_list, prefix="valid ")
    # statstics_calculator.plot_all()
    # dynamics_calculator.add_traj(traj_true, traj_pred)
    # dynamics_calculator.plot_all()

    # TEST set
    traj_pred, traj_true, time_list = compute_traj(
        trained_model, true_model, args.init_pos_test, nb_steps
    )
    np.save(os.path.join(save_dir_path, "traj_pred_test.npy"), traj_pred)
    np.save(os.path.join(save_dir_path, "traj_true_test.npy"), traj_true)
    np.save(os.path.join(save_dir_path, "time_list_test.npy"), time_list)

    statstics_calculator.add_traj(traj_true, traj_pred, time_list, prefix="test ")
    statstics_calculator.plot_all()
    dynamics_calculator.add_traj(traj_true, traj_pred)
    dynamics_calculator.plot_all()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_iter_train", type=int, default=1000)
    parser.add_argument("--n_iter_valid", type=int, default=1000)
    parser.add_argument("--n_iter_test", type=int, default=1000)
    parser.add_argument("--init_pos_train", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    parser.add_argument("--init_pos_valid", nargs="+", type=float, default=[0.9, -0.6, 0.028])
    parser.add_argument("--init_pos_test", nargs="+", type=float, default=[0.01, 2.5, 3.07])
    parser.add_argument("--delta_t", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--project", type=str, default="rossler")

    args = parser.parse_args()
    main(args)
