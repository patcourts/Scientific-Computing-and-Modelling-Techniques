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