####
# This code is from MSc lectures
#
####
import numpy as np

class LinearBasis:
    """
    Represents a 1D linear basis.
    """
    def __init__(self):
        self.num_basis = 2 # The number of basis functions
        
    def __call__(self, x):
        """
        x : 1D array of inputs
        """
        return [1., x[0]]
    

class PolynomialBasis:
    """
    A set of polynomial basis functions.
    
    Arguments:
    degree  -  The degree of the polynomial.
    """
    def __init__(self, degree):
        self.degree = degree
        self.num_basis = degree + 1
    
    def __call__(self, x):
        return np.array([x[0] ** i for i in range(self.degree + 1)])
    

class RadialBasisFunctions:
    """
    A set of radial basis functions.
    
    Arguments:
    X   -  The centers of the radial basis functions.
    ell -  The assumed lengthscale.
    """
    def __init__(self, X, ell):
        self.X = X
        self.ell = ell
        self.num_basis = X.shape[0]

    def __call__(self, x):
        return np.exp(-.5 * (x - self.X) ** 2 / self.ell ** 2).flatten()

    

def design_matrix(X, phi):
    """
    Arguments:
    
    X: The observed inputs
    phi:  The basis functions
    """
    num_observations = X.shape[0]
    num_basis = phi.num_basis
    Phi = np.zeros((num_observations, num_basis))
    for i in range(num_observations):
        Phi[i, :] = phi(X[i, :])
    return Phi

def least_squares_MLE(Phi, Y, X):
    """Compute maximum likelihood estimate of mean and standard deviation of weights"""
    w_MLE, res_MLE, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    sigma_MLE = np.sqrt(res_MLE / X.shape[0])
    return w_MLE, sigma_MLE