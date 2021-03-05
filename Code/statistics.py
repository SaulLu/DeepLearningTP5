import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, qr, solve
from scipy.linalg import expm


def plot3D_traj(traj_pred, traj_true):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], "-.", label="true")
    ax.plot(traj_pred[:, 0], traj_pred[:, 1], traj_pred[:, 2], "--", label="pred")
    ax.legend()
    return ax, fig


def plot_x_traj(traj_pred, traj_true, time_list):
    return plot_i_traj(traj_pred, traj_true, time_list, 0)


def plot_y_traj(traj_pred, traj_true, time_list):
    return plot_i_traj(traj_pred, traj_true, time_list, 1)


def plot_z_traj(traj_pred, traj_true, time_list):
    return plot_i_traj(traj_pred, traj_true, time_list, 2)


def plot_i_traj(traj_pred, traj_true, time_list, idx):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(time_list, traj_true[:, idx], "-.", label="true")
    ax.plot(time_list, traj_pred[:, idx], "-.", label="pred")
    ax.legend()
    return ax, fig


def plot_x_pdf(traj_pred, traj_true, time_list):
    return plot_i_pdf(traj_pred, traj_true, time_list, 0)


def plot_y_pdf(traj_pred, traj_true, time_list):
    return plot_i_pdf(traj_pred, traj_true, time_list, 1)


def plot_z_pdf(traj_pred, traj_true, time_list):
    return plot_i_pdf(traj_pred, traj_true, time_list, 2)


def plot_i_pdf(traj_pred, traj_true, time_list, idx):
    fig = plt.figure()
    ax = fig.gca()

    plt.hist(traj_true[:, idx], label="true", alpha=0.8)
    plt.hist(traj_pred[:, idx], label="pred", alpha=0.8)
    ax.legend()
    return ax, fig


def autocorrelation():
    # TODO
    pass


##### Following metrics are dynamics not statistics !!


def lyapunov_exponent(traj, jacobian, max_it=1000, delta_t=1e-3, mode="discrete"):

    n = traj.shape[1]
    w = np.eye(n)
    rs = []
    chk = 0

    for i in range(max_it):
        jacob = jacobian(traj[i, :])

        if mode == "discrete":
            w_next = np.dot(jacob, w)
        elif mode == "continuous":
            # WARNING this is true for the jacobian of the continuous system!
            w_next = np.dot(expm(jacob * delta_t), w)
            # if delta_t is small you can use:
            # w_next = np.dot(np.eye(n)+jacob * delta_t,w)
        else:
            raise NotImplementedError()

        w_next, r_next = qr(w_next)

        # qr computation from numpy allows negative values in the diagonal
        # Next three lines to have only positive values in the diagonal
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)
        w = w_next
        if i // (max_it / 100) > chk:
            print(i // (max_it / 100))
            chk += 1

    return np.mean(np.log(rs), axis=0) / delta_t


def newton(f, jacob, x):
    # newton raphson method
    tol = 1
    compt_max = 10000000
    compt = 0
    while tol > 1e-5 and compt < compt_max:
        # WARNING this is true for the jacobian of the continuous system!
        tol = x
        x = x - solve(jacob(x), f(x))
        tol = norm(tol - x)
        compt += 1
    return x