import matplotlib.pyplot as plt


def plot3D_traj(traj_pred, traj_true):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(traj_pred[:, 0], traj_pred[:, 1], traj_pred[:, 2], label="pred")
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], label="true")
    ax.legend()
    return ax, fig