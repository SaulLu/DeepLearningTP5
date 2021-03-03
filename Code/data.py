import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rossler_map import RosslerMap


class RosslerAttractorDataset(Dataset):
    def __init__(
        self, n_iter: int = 2000000, delta_t: float = 1e-2, init_pos=np.array([-5.75, -1.6, 0.02])
    ):
        super().__init__()
        rossler_map = RosslerMap(delta_t=delta_t)
        self.traj, _ = rossler_map.full_traj(n_iter, init_pos)
        self.traj = torch.tensor(self.traj, dtype=torch.float)

    def __getitem__(self, index):
        return tuple(self.traj[index], self.traj[index + 1])

    def __len__(self):
        return self.traj.size(0) - 1  # Last item don't have true_y


class RosslerAttractorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_iter_train: int = 2000000,
        n_iter_valid: int = 20000,
        n_iter_test: int = 2000000,
        delta_t: float = 1e-2,
        init_pos_train=np.array([-5.75, -1.6, 0.02]),
        init_pos_test=np.array([-5.70, -1.5, -0.02]),
        batch_size: int = 32,
    ):
        super().__init__()
        self.n_iter_train = n_iter_train
        self.n_iter_valid = n_iter_valid
        self.n_iter_test = n_iter_test
        self.delta_t = delta_t
        self.init_pos_train = init_pos_train
        self.init_pos_test = init_pos_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = RosslerAttractorDataset(
                self.n_iter_train + self.n_iter_valid, self.delta_t, self.init_pos_train
            )
            self.dataset_train, self.dataset_valid = random_split(
                dataset, [self.n_iter_train, self.n_iter_valid]
            )
        if stage == "test" or stage is None:
            self.dataset_test = RosslerAttractorDataset(
                self.n_iter_test, self.delta_t, self.init_pos_test
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)