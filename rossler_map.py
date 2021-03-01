import numpy as np
from scipy.integrate import solve_ivp


class RosslerMap:
    """
    Rossler attractor
    With a=0.2, b=0.2, and c=5.7
    """

    def __init__(_, a=0.2, b=0.2, c=5.7, delta_t=1e-3):
        _.a, _.b, _.c = a, b, c
        _.delta_t = delta_t

    def v_eq(_, t=None, v=None):
        x, y, z = v[0], v[1], v[2]
        dot_x = -y - z
        dot_y = x + _.a*y
        dot_z = _.b + z*(x-_.c)
        return np.array([dot_x, dot_y, dot_z])

    def jacobian(_, v):
        x, z = v[0], v[2]
        res = np.array([[       0,      -1,       -1],
                       [        1,     _.a,        0],
                       [        z,       0,   x-_.c]])
        return res

    def full_traj(_, nb_steps, init_pos):
        t = np.linspace(0, nb_steps * _.delta_t, nb_steps)
        f = solve_ivp(_.v_eq, [0, nb_steps * _.delta_t], init_pos, method='RK45', t_eval=t)
        return np.moveaxis(f.y, -1, 0),t
    
    def equilibrium(_):
        x0 = (_.c-np.sqrt(_.c**2-4*_.a*_.b))/2
        y0 = (-_.c+np.sqrt(_.c**2-4*_.a*_.b))/(2*_.a)
        z0 = (_.c-np.sqrt(_.c**2-4*_.a*_.b))/(2*_.a)
        return np.array([x0,y0,z0])
