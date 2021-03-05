import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm, qr, solve
from scipy.linalg import expm

from rossler_map import RosslerMap


def lyapunov_exponent(traj, jacobian, max_it=1000, delta_t=1e-3):

    n = traj.shape[1]
    w = np.eye(n)
    rs = []
    chk = 0

    for i in range(max_it):
        jacob = jacobian(traj[i, :])
        # WARNING this is true for the jacobian of the continuous system!
        w_next = np.dot(expm(jacob * delta_t), w)
        # if delta_t is small you can use:
        # w_next = np.dot(np.eye(n)+jacob * delta_t,w)

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
    while tol > 1e-5:
        # WARNING this is true for the jacobian of the continuous system!
        tol = x
        x = x - solve(jacob(x), f(v=x))
        tol = norm(tol - x)
    return x


if __name__ == "__main__":

    Niter = 2000000
    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6, 0.02])
    traj, t = ROSSLER_MAP.full_traj(Niter, INIT)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])

    fix_point = newton(ROSSLER_MAP.v_eq, ROSSLER_MAP.jacobian, INIT)

    error = norm(fix_point - ROSSLER_MAP.equilibrium())
    print("equilibrium state :", fix_point, ", error : ", error)

    # lyap = lyapunov_exponent(traj, ROSSLER_MAP.jacobian, max_it=Niter, delta_t=delta_t)
    # print("Lyapunov Exponents :", lyap, "with delta t =", delta_t)

    plt.show()
