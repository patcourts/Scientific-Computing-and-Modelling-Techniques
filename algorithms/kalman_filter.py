import numpy as np

#kalman filter split into predict and then update sections, predictions are made and then when measurements are taken, values 
#are updated
def kalman_predict(m, cov, A, Q, t):
    predicted_m = np.linalg.solve((np.identity(2) - t*A), m)
    predicted_cov = np.linalg.inv((np.identity(2) - t*A))@cov@(np.linalg.inv((np.identity(2) - t*A)).T) + 2*t*Q
    return predicted_m, predicted_cov

def kalman_gain(cov, H, R):
    return cov@(H.T)@(np.linalg.inv(R+H@cov@(H.T)))# np.linalg.solve would not work BECAUSE IS NOT AN INVERSE PROBLEM IS MATRIX CALCULATION


def kalman_update(m, K, cov, H, obs):
    new_m = m - K@(H@m - obs)
    new_cov = cov - K@H@cov
    return new_m, new_cov


def linear_kalman_filter(initial_m, initial_C, A, Q, time, step_size, H, R, observations):
    num_steps = int(time/step_size)

    m_estimates = np.empty((num_steps, 2))
    C_estimates = np.empty((num_steps, 2, 2))

    m_estimates[0] = initial_m
    C_estimates[0] = initial_C

    KF_counter = 0

    for i in range(1, num_steps):#from range 1 as already have initial values
        predicted_m, predicted_cov = kalman_predict(m_estimates[i-1], C_estimates[i-1], A, Q, step_size)
        if KF_counter > 0 and KF_counter % 10 == 0: #measurments occur every 10 timesteps
            gain = kalman_gain(predicted_cov, H, R)
            m_estimates[i], C_estimates[i] = kalman_update(predicted_m, gain, predicted_cov, H, observations[i//10])
        else: #if not measurement just append predicted value
            m_estimates[i] = predicted_m
            C_estimates[i] = predicted_cov
        KF_counter += 1

    position_std_dev = np.zeros(len(m_estimates))
    velocity_std_dev = np.zeros(len(m_estimates))
    for i in range(len(C_estimates)): # calculating std from covariance matrices
        position_std_dev[i] = (np.sqrt(C_estimates[i, 0, 0]))
        velocity_std_dev[i] = (np.sqrt(C_estimates[i, 1, 1]))
        
    return m_estimates, position_std_dev, velocity_std_dev
