from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data import RosslerAttractorDataModule
from model import Model


def main(args):
    wandb_logger = WandbLogger(project=args.project)

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        save_best_only=True,
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
        batch_size=args.batch_size,
    )

    model = Model(lr=args.lr)

    trainer = Trainer(
        gpus=args.gpus,
        logger=wandb_logger,
        auto_lr_find=True,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.tune(model)

    trainer.fit(model, datamodule)

    trainer.test(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_iter_train", type=int, default=2000000)
    parser.add_argument("--n_iter_valid", type=int, default=20000)
    parser.add_argument("--n_iter_test", type=int, default=2000000)
    parser.add_argument("--init_pos_train", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    parser.add_argument("--init_pos_test", nargs="+", type=float, default=[-5.70, -1.5, -0.02])

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--delta_t", type=float, default=1e-2)

    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--gpus", default=None)

    args = parser.parse_args()
    main(args)