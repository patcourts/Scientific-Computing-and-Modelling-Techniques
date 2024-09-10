from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

def prior(alpha, N):
    """Compute mean and covariance matrices of the weight prior"""
    m0 = np.zeros(N)
    S0 = 1./alpha *np.eye(N)
    return m0, S0

def posterior(Phi, y, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(y)
    m_N = m_N
    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N

def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N).ravel()
    # Only compute variances (diagonal elements of covariance matrix)
    y_epi = np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    y_var = 1/beta + y_epi   
    return y, y_epi, y_var

def plot_posterior(m_N, s_N, a_0, a_1):
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(w0.shape + (2,))
    pos[:, :, 0] = w0
    pos[:, :, 1] = w1
    
    posterior = multivariate_normal(m_N.ravel(), s_N)
    plt.contourf(w0, w1, posterior.pdf(pos))
    plt.plot([a_0], [a_1], 'rx', label='Truth')
    plt.axis('equal')
    plt.xlabel('$w_0$')
    plt.ylabel('$w_1$')
    
def plot_data(X, y):
    plt.plot(X[:,0], y[:,0], 'kx', ms=10)   
    
def plot_truth(X, y, label='Truth'):
    plt.plot(X[:,0], y[:,0], 'k--', label=label)
    
def plot_posterior_samples(X, y):
    plt.plot(X, y, 'r-')
    plt.axis('equal')    
    
def plot_predictive(X, y, y_epi, y_var):
    sigma_epi = np.sqrt(y_epi) # epistemitic uncertainty
    sigma_tot = np.sqrt(y_var) # total uncertainty
    
    y_el = y - 2*sigma_epi
    y_tl = y - 2*sigma_tot
    y_eu = y + 2*sigma_epi
    y_tu = y + 2*sigma_tot
    
    plt.plot(X[:,0], y, 'b-', label='Prediction')
    plt.fill_between(X[:,0], y_el, y_eu, color='C2',
                     label='Epistemic uncertainty', alpha=0.3)    
    plt.fill_between(X[:,0], y_tl, y_el, color='C1',
                     label='Total uncertainty', alpha=0.3)
    plt.fill_between(X[:,0], y_eu, y_tu, color='C1',
                     alpha=0.3)