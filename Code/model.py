import pytorch_lightning as pl
import torch
import torch.nn as nn
from jacobian import JacobianReg


class Model(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 10,
        in_size: int = 3,
        out_size: int = 3,
        lr: float = 1e-3,
        lambda_jr=0.01,  # lambda jacobian regularisation
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lr = lr
        self.lambda_jr = lambda_jr

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
        return x + self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data.requires_grad = True  # this is essential!

        output = self(data)
        mse = self.criterion(output, target)
        print(f"data {data.requires_grad}")
        print(f"output {output.requires_grad}")
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
