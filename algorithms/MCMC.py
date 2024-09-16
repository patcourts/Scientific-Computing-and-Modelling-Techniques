import numpy as np

def metropolis_hastings(x0, rho, L, N, step_size, adapt_step_size=False):
    x = [x0]
    log_acceptance_probability = []
    acceptance_counter = 0
    for i in range(1, N):
        x_candidate = x[i-1] + step_size**2 * L@np.random.normal(size = len(L))
        log_acceptance_ratio = min(0, rho(x_candidate) - rho(x[i-1])) #log version used again to avoid computational problems
        log_acceptance_probability.append(log_acceptance_ratio)
        log_z = np.log(np.random.uniform())#compared with logged value
        #if meets acceptance allow the proposed move
        if log_z <= log_acceptance_ratio:
            x.append(x_candidate)
            acceptance_counter += 1 #easy to calculate probability of acceptance
        #if not then stay where is
        else:
            x.append(x[i-1])
        
        #modifying step size at every 100th iteration
        if adapt_step_size:
            if i % 100 == 0:
                av_acc_prob = (1/i)*np.sum(np.exp(log_acceptance_probability)) #calculating average of the acceptance probabilites so far
                if av_acc_prob > 0.27:
                    step_size = step_size *2
                elif av_acc_prob < 0.19:
                    step_size = step_size /2
    return x, log_acceptance_probability, acceptance_counter

def pCN_MH_MCMC(potential, N, u0, cov, step_size, k_max, y):
    #initiating lists/arrays and setting initial values
    u = np.zeros(shape = (N, k_max))
    u[0] = u0
    alpha_hist = [0.0, ]
    Hu_0 = get_Hu(k_max, u0)
    pot_hist = [potential(Hu_0, y),] 
    acceptance_counter = 0
    #iterating over number of iterations
    for i in range(1, N):
        assert 0.0 <= step_size <= 1.0 #for pCN have step size restriction

        u_proposal = ((1-step_size**2)**0.5)*u[i-1]+step_size*cov.dot(np.random.normal(size = u0.shape))
        
        pot_u = pot_hist[-1] #take latest value of potential
        pot_u_proposal = potential(get_Hu(k_max, u_proposal), y) #calculate new value of potential

        #accept / reject
        log_alpha = min(0, pot_u - pot_u_proposal)
        log_z = np.log(np.random.random())
        alpha_hist.append(np.exp(log_alpha))
        if log_z <= log_alpha:
            acceptance_counter += 1
            u[i] = (u_proposal)
            pot_hist.append(pot_u_proposal)
            
        else:
            u[i] = u[i-1]
            pot_hist.append(pot_u)
    return u, acceptance_counter, alpha_hist


#functions to evaluate montecarlo models taken/adapted from week 5 lab

#calculates the average jump size
def average_jump(u):
    T = u.shape[0]
    return sum(np.linalg.norm(u[i,:] - u[i-1,:]) for i in range(T)[1:]) / T

#calculates the correlation within the trace sequence from MH MCMC, ideally want a low autocorrelation, i.e. the sequence
#should not be highly correlated with itself
def autocorr(seq, lag=0):
    assert len(seq.shape) == 1
    assert lag >= 0
    N = seq.shape[0]
    seq1 = seq[0:N-lag]
    seq2 = seq[lag:N]
    m1 = np.average(seq1)
    m2 = np.average(seq2)
    seq1c = seq1 - m1 * np.ones(seq1.shape)
    seq2c = seq2 - m2 * np.ones(seq2.shape)
    sigma11 = np.sum(seq1c * seq1c)
    sigma22 = np.sum(seq2c * seq2c)
    sigma12 = np.sum(seq1c * seq2c)
    if sigma11 == 0.0 or sigma22 == 0.0:
        return 1.0
    else:
        return sigma12 / np.sqrt(sigma11 * sigma22)

#calculates ess which gives a representation of the independancy of the series, i.e. if = 1 then evry point independant
#of every other point
def effective_sample_size_ratio(seq, lags):
    return 1.0 / (1.0 + 2.0 * np.sum([autocorr(seq, l) for l in lags if l >= 1]))