import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rossler_map import RosslerMap


class DiscreteRosslerAttractorDataset(Dataset):
    def __init__(
        self, n_iter: int = 2000000, delta_t: float = 1e-2, init_pos=np.array([-5.75, -1.6, 0.02])
    ):
        super().__init__()
        if isinstance(init_pos, tuple) or isinstance(init_pos, list):
            init_pos = np.array(init_pos)
        rossler_map = RosslerMap(delta_t=delta_t)
        self.traj, _ = rossler_map.full_traj(n_iter, init_pos)
        self.traj_n_1 = torch.tensor(self.traj, dtype=torch.float)
        self.traj_n_2 = torch.tensor(self.traj.copy(), dtype=torch.float)

    def __getitem__(self, index):
        return tuple([self.traj_n_1[index], self.traj_n_2[index + 1]])

    def __len__(self):
        return self.traj_n_1.size(0) - 1  # Last item don't have true_y


class DiscreteRosslerAttractorDataModule(pl.LightningDataModule):
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
        self.n_iter_train = n_iter_train
        self.n_iter_valid = n_iter_valid
        self.n_iter_test = n_iter_test
        self.delta_t = delta_t
        self.init_pos_train = init_pos_train
        self.init_pos_valid = init_pos_valid
        self.init_pos_test = init_pos_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        RosslerAttractorDataset = DiscreteRosslerAttractorDataset
        if stage == "fit" or stage is None:
            self.dataset_train = RosslerAttractorDataset(
                self.n_iter_train, self.delta_t, self.init_pos_train
            )
            self.dataset_valid = RosslerAttractorDataset(
                self.n_iter_valid,
                self.delta_t,
                self.init_pos_valid,
            )
        if stage == "test" or stage is None:
            self.dataset_test = RosslerAttractorDataset(
                self.n_iter_test,
                self.delta_t,
                self.init_pos_test,
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)


class ContinuousRosslerAttractorDataset(Dataset):
    def __init__(
        self,
        n_iter: int = 200,
        delta_t: float = 1e-2,
        init_positions=[np.array([-5.75, -1.6, 0.02]), np.array([-3.75, 0.02, 2.9])],
        n_samples=800,
    ):
        super().__init__()
        self.delta_t = delta_t
        self.trajs = []
        self.ts = []
        rossler_map = RosslerMap(delta_t=delta_t)

        for init_pos in init_positions:
            init_pos_temp = init_pos
            for i in range(n_samples):
                if i != 0:
                    init_pos_temp = traj[-1, :]
                traj, t = rossler_map.full_traj(n_iter, init_pos_temp)
                traj = torch.tensor(traj, dtype=torch.float)
                t = torch.tensor(t, dtype=torch.float)
                # step = t.size(0) // n_div
                # ruptures = [i * step for i in range(n_div)]
                # ruptures.append(t.size(0))
                # for idx in range(n_div):
                #     self.trajs.append(traj[ruptures[idx] : ruptures[idx + 1]])
                #     self.ts.append(t[ruptures[idx] : ruptures[idx + 1]])

                self.trajs.append(traj)
                self.ts.append(t)

    def __getitem__(self, index):
        return self.trajs[index], self.ts[index]

    def __len__(self):
        return len(self.trajs)


class ContinuousRosslerAttractorDatasetOld(Dataset):
    def __init__(
        self, n_iter: int = 2000000, delta_t: float = 1e-2, init_pos=np.array([-5.75, -1.6, 0.02])
    ):
        super().__init__()
        self.delta_t = delta_t
        rossler_map = RosslerMap(delta_t=delta_t)
        self.traj, self.t = rossler_map.full_traj(n_iter, init_pos)
        self.traj = torch.tensor(self.traj, dtype=torch.float)
        self.steps = [1, 2, 3, 4, 5, 6]

    def __getitem__(self, index):
        new_index = index // len(self.steps)
        step = self.steps[index % len(self.steps)]
        return tuple([self.traj[new_index], step * self.delta_t, self.traj[new_index + step]])

    def __len__(self):
        return len(self.steps) * self.traj.size(0) - np.sum(
            self.steps
        )  # Last item don't have true_y


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
        self.n_iter_train = n_iter_train
        self.n_iter_valid = n_iter_valid
        self.n_iter_test = n_iter_test
        self.delta_t = delta_t
        self.init_pos_train = init_pos_train
        self.init_pos_valid = init_pos_valid
        self.init_pos_test = init_pos_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        RosslerAttractorDataset = ContinuousRosslerAttractorDataset
        if stage == "fit" or stage is None:
            self.dataset_train = RosslerAttractorDataset(
                self.n_iter_train,
                self.delta_t,
                # self.init_pos_train
            )
            self.dataset_valid = RosslerAttractorDataset(
                self.n_iter_valid,
                self.delta_t,
                # self.init_pos_valid
                n_samples=200,
            )
        if stage == "test" or stage is None:
            self.dataset_test = RosslerAttractorDataset(
                self.n_iter_test,
                self.delta_t,
                # self.init_pos_test
                n_samples=200,
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)