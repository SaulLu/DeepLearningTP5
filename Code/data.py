import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from rossler_map import RosslerMap


class RosslerAttractorDataset(Dataset):
    def __init__(
        self,
        delta_t: float = 1e-2,
        n_iter: int = 1e7 + 1,
        init_pos=np.array([-5.75, -1.6, 0.02]),
        mean=None,
        std=None,
    ):
        super().__init__()
        if isinstance(init_pos, tuple) or isinstance(init_pos, list):
            init_pos = np.array(init_pos)
        rossler_map = RosslerMap(delta_t=delta_t)
        self.traj, _ = rossler_map.full_traj(n_iter, init_pos)

        self.mean = mean if mean is not None else self.traj.mean(axis=0)
        self.std = std if mean is not None else self.traj.std(axis=0)
        self.traj -= self.mean
        self.traj /= self.std
        self.traj_n_1 = torch.tensor(self.traj, dtype=torch.float)
        self.traj_n_2 = torch.tensor(self.traj.copy(), dtype=torch.float)

    def __getitem__(self, index):
        return tuple([self.traj_n_1[index], self.traj_n_2[index + 1]])

    def __len__(self):
        return self.traj_n_1.size(0) - 1  # Last item don't have true_y


class RosslerAttractorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_iter_train: int = 2000000,
        n_iter_valid: int = 20000,
        n_iter_test: int = 2000000,
        delta_t: float = 1e-2,
        init_pos_train=np.array([-5.75, -1.6, 0.02]),
        init_pos_valid=np.array([0.01, 2.5, 3.07]),
        init_pos_test=np.array([-5.70, -1.5, -0.02]),
        batch_size: int = 32,
    ):
        super().__init__()
        self.iterations = [n_iter_train, n_iter_valid, n_iter_test]
        self.positions = [init_pos_train, init_pos_valid, init_pos_test]
        self.delta_t = delta_t
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset_train = RosslerAttractorDataset(
            self.delta_t, self.iterations[0], self.positions[0]
        )
        mean = self.dataset_train.mean
        std = self.dataset_train.std
        self.dataset_valid = RosslerAttractorDataset(
            self.delta_t, self.iterations[1], self.positions[1], mean, std
        )
        self.dataset_test = RosslerAttractorDataset(
            self.delta_t, self.iterations[2], self.positions[2], mean, std
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
