import argparse

import numpy as np

# from scipy.interpolate import interp1d

INITIAL_CONDITION = [-5.75, -1.6, 0.02]
TRAJECTORY_DUR = 10000


class Rossler_model:
    def __init__(self, model_cls, checkpoint_path):
        self.rosler_nn = model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.nb_steps = int(
            TRAJECTORY_DUR // self.rosler_nn.hparams.delta_t
        )  # int(10000 // self.delta_t)
        self.rosler_nn.normalize = True
        self.initial_condition = np.array(INITIAL_CONDITION)

    def full_traj(self, initial_condition=np.array(INITIAL_CONDITION), y_only=True):
        traj, t = self.rosler_nn.full_traj(self.nb_steps, initial_condition)
        if y_only:
            traj = traj[:, 1]
        # TODO: warning interpolate trajectories is not done yet
        return traj

    def save_traj(self, y):
        np.save("traj.npy", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", nargs="+", type=float, default=[-5.75, -1.6, 0.02])
    value = parser.parse_args()
    ROSSLER = Rossler_model()
    y = ROSSLER.full_traj()
    ROSSLER.save_traj(y)
