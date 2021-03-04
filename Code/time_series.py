import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

from model import Model

INITIAL_CONDITION = [-5.75, -1.6, 0.02]
TRAJECTORY_DUR = 10


class Rossler_model:
    def __init__(self, delta_t, model_cls, checkpoint_path):
        self.delta_t = delta_t  # if discrete model your delta_t
        # if continuous model chose one <=1e-2
        self.nb_steps = int(TRAJECTORY_DUR // self.delta_t)  # int(10000 // self.delta_t)

        self.rosler_nn = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.rosler_nn.eval()
        self.initial_condition = np.array(INITIAL_CONDITION)

    def full_traj(self, initial_condition=np.array(INITIAL_CONDITION), y_only=True):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.

        traj = [initial_condition[np.newaxis, :]]
        t = [self.delta_t]
        # print(f"self.nb_steps: {self.nb_steps}")
        with torch.no_grad():
            for _ in range(self.nb_steps - 1):
                new_coord = self.rosler_nn(torch.tensor(traj[-1], dtype=torch.float)).numpy()
                # print(f"new_coord: {new_coord.shape}")
                traj.append(new_coord)
                t.append(t[-1] + self.delta_t)

        # print(f"traj[-1]: {traj[-1].shape}")

        traj = np.concatenate(traj, axis=0)
        t = np.array(t)
        # print(f"traj: {traj.shape}")
        if y_only:
            traj = traj[:, 1]

        ## TODODOODODOD
        # print(f"traj: {traj.shape}")
        # print(f"t: {t.shape}")

        # t_new = np.linspace(0, TRAJECTORY_DUR, int(TRAJECTORY_DUR // 1e-2))
        # print(f"t_new: {t_new.shape}")

        # # # if your delta_t is different to 1e-2 then interpolate y
        # # # in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, t, traj)
        # print(f"y_new: {y_new.shape}")
        # # I expect that y.shape = (1000000,)
        return traj

    def save_traj(self, y):
        # save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)

        np.save("traj.npy", y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--init", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    value = parser.parse_args()

    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)
