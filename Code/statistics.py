import matplotlib.pyplot as plt


def plot3D_traj(traj):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
    return ax, fig