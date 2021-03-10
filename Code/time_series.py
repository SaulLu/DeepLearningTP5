import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

INITIAL_CONDITION = [-5.75, -1.6, 0.02]
TRAJECTORY_DUR = 10000


class Rossler_model:
    def __init__(self, delta_t, model_cls, checkpoint_path):
        self.delta_t = delta_t 
        self.nb_steps = int(TRAJECTORY_DUR // self.delta_t)  # int(10000 // self.delta_t)
        self.rosler_nn = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.rosler_nn.normalize = True
        self.initial_condition = np.array(INITIAL_CONDITION)

    def full_traj(self, initial_condition=np.array(INITIAL_CONDITION), y_only=True):
        traj, t = self.rosler_nn.full_traj(TRAJECTORY_DUR, initial_condition)
        if y_only:
            traj = traj[:, 1]
        return traj

    def save_traj(self, y):
        np.save("traj.npy", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    value = parser.parse_args()
    ROSSLER = Rossler_model(delta_t)
    y = ROSSLER.full_traj()
    ROSSLER.save_traj(y)
