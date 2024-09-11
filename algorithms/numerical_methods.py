import numpy as np


def ImplicitEuler(x0, time, dt, A):
    """
    Implicit Euler methos to solve ODEs

    x0: initial state, vector size of dimensions
    time: total time of simulation (s) int
    dt: time step (s) float
    A: state transition matrix

    returns array of states
    """
    num_steps = int(time/dt)
    x = np.zeros(shape=(len(x0), num_steps))
    x[:, 0] = x0
    for i in range(0, num_steps-1):
        x[:, i+1] = np.linalg.solve((np.identity(len(x0)) - dt*A), x[:, i])
    return x