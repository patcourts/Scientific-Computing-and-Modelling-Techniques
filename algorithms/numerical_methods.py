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


def ExplicitEuler(f, times, x0):
    x = [x0,]
    for i in range(len(times))[1:]:
        x_old = x[-1]
        x_new = x_old + (times[i] - times[i-1]) * f(x_old)
        x.append(x_new)
    return x



#numerical integration
def traps(a, b, num_steps, func):
    '''function that takes integration limits, number of points of integration and the function to be integrated and returns
    the trapzoidal approximation of the integral of the function using equations from MA934 assignment 3.
    '''
    h = (b-a)/(num_steps)
    a1 = (func(a) + func(b))/2
    asum = 0
    for i in range(1, num_steps):
        asum += func(a+i*h)
    return (a1 + asum) * h 

def simpson(a, b, num_steps, func):
    '''function that takes integration limits, number of points of integration and the function to be integrated and returns
    the simpsons 1/3 approximation of the integral of the function using equations from MA934 assignment 3.
    '''
    #simpsons third only accuarte for even numbers of step size
    if num_steps%2 == 0:
        h = (b-a)/(num_steps)
        sum1 = 0
        sum2 = 0
        for i in range(1, (num_steps//2) +1):
            sum1 += func(a+((2*i)-1)*h)
        for i in range(1, num_steps//2):
            sum2 += func(a + (2*i)*h)
        return (1/3)*h*(func(a)+func(b)+ 4*sum1 + 2*sum2)
    return None
    
    
def simpson2(a, b, num_steps, func):
    '''function that takes integration limits, number of points of integration and the function to be integrated and returns
    the simpson 3/8 approximation of the integral of the function using equations from MA934 assignment 3.
    '''
    #only works for mulitples of three
    if num_steps%3 == 0:
        h = (b-a)/(num_steps)
        sum1 = 0
        sum2 = 0
        for i in range(1, int(num_steps)):
            if i%3 != 0:
                sum1 += func(a + (i*h))
        for i in range(1, (int(num_steps/3))):
            sum2 += func(a + (3*i*h))
        return (3/8)*h*(func(a)+func(b)+3*sum1 + 2*sum2)
    return None
    