# this code is for the project 'Identification of Nonlinear Dynamic Panels under Partial Stationarity', by Wayne Gao and Rui Wang
# it is implemneting CLR inference method to compute confidence interval for the static ordered choice model

import numpy as np
from scipy.stats import multivariate_normal, norm

# random seed
np.random.seed(123)

# Simulation configurations
alpha = 0.05                # Significance level
T = 2                       # Time period
theta0 = np.array([1, 1])   # True parameter
N = 2000                    # Sample size

sigma_z = 1.0               # Standard deviation of Z
mu = np.zeros(2)            # Mean
rho = 0.25                  # Correlation coefficient
sigma = np.array([[1, rho], [rho, 1]])  # Covariance matrix

R = 1000                    # Number of simulation repetitions
l_est = np.zeros(R)         # Lower bounds
u_est = np.zeros(R)         # Upper bounds
B_size = 1000               # Simulation draws for calculating the CLR test critical value
rndgrid = 1000              # Number of randomly drawn grid points
tuning = 0.1 / np.log(N)    # Tuning parameter

u = np.random.randn(N, B_size)  # Random multipliers used in the CLR test implementation
test_b = np.zeros(B_size)       # Simulated version of the test statistic
b_c = 2.65                      # Bandwidth scale

# Define kernel function
def kern(v):
    return (15/16) * ((1 - v**2)**2) * (abs(v) <= 1)

v_hat_CLR = np.zeros(rndgrid)   # v_hat stores the indices of v in the estimated contact set
wsize_CLR = b_c / (N ** (1/9))  # Undersmoothing for d=4

## Main simulation loop
for r in range(R):

    # Generate the true covariate
    Z1 = norm.rvs(loc=0, scale=sigma_z, size=(N, T))
    Z2 = norm.rvs(loc=0, scale=sigma_z, size=(N, T))
    Z = np.hstack((Z1, Z2))  

    # Generate grid for covariates
    Zgrid1 = norm.rvs(loc=0, scale=sigma_z, size=(rndgrid, T))
    Zgrid2 = norm.rvs(loc=0, scale=sigma_z, size=(rndgrid, T))
    Zgrid = np.hstack((Zgrid1, Zgrid2))  

    # Generate error term
    a = (Z1 + Z2) @ theta0 / (4 * T * sigma_z)  # fixed effect
    ep = multivariate_normal.rvs(mean=mu, cov=sigma, size=N)  # time changing error

    # Dependent variable
    b2 = -1       # threshold of ordered choice
    b3 = 1
    Ystar = np.array([ep[:, 0] + a + Z1 @ theta0, ep[:, 1] + a + Z2 @ theta0]).T  # latent dependent variable
    Y = 1 * (Ystar <= b2) + 2 * ((Ystar > b2) & (Ystar <= b3)) + 3 * (Ystar > b3)  # observed dependent variable
    
    # Define conditional moment function within the simulation loop
    def cmom(theta):
        # Covariate index
        covind = np.column_stack((Z1 @ theta, Z2 @ theta))
        
        # Define four moment functions
        g1 = np.zeros((N, T, T))
        g2 = np.zeros((N, T, T))
        g3 = np.zeros((N, T, T))
        g4 = np.zeros((N, T, T))
        
        for s in range(T):
            for t in range(T):
                if s != t:
                    g1[:, s, t] = (covind[:, t] - covind[:, s] >= 0).astype(int) * ((Y[:, s] == 1).astype(int) - (Y[:, t] == 1).astype(int))
                    g2[:, s, t] = (b2 - covind[:, s] >= b3 - covind[:, t]).astype(int) * ((Y[:, s] == 1).astype(int) - (Y[:, t] <= 2).astype(int))
                    g3[:, s, t] = (b3 - covind[:, s] >= b2 - covind[:, t]).astype(int) * ((Y[:, s] <= 2).astype(int) - (Y[:, t] == 1).astype(int))
                    g4[:, s, t] = (covind[:, t] - covind[:, s] >= 0).astype(int) * ((Y[:, s] <= 2).astype(int) - (Y[:, t] <= 2).astype(int))
            
        # Concatenate all the relevant moment functions for returning
        g = np.column_stack([g1[:, 0, 1], g1[:, 1, 0], g2[:, 0, 1], g2[:, 1, 0], g3[:, 0, 1], g3[:, 1, 0], g4[:, 0, 1], g4[:, 1, 0]])
        return g
    
    # Calculate kernel values
    hsize_CLR = wsize_CLR * np.std(Z, axis=0)  # bandwidth
    kern_values_CLR = np.ones((N, rndgrid))    # initial kernel values at grid
    temp_k = np.ones((N, N))
    
    for k2 in range(Z.shape[1]): 
        kern_values_CLR *= kern((Zgrid[:, k2][:, None] - Z[:, k2][None, :]).T / hsize_CLR[k2])
        temp_k *= kern((Z[:, k2][:, None] - Z[:, k2][None, :]).T / hsize_CLR[k2])
        
    # Define the test function
    def test(theta):
        g = cmom(theta)  # conditional moment
        dg = g.shape[1]  # dimension of moments

        u_hat_CLR = np.zeros((N, dg))           # residual of moment
        temp3 = np.zeros((N, rndgrid, dg))      # uhat*kern
        sigma_hat_CLR = np.zeros((rndgrid, dg)) # standard deviation
        std_m_hat_CLR = np.zeros((rndgrid, dg)) # t statistic
        w_hat_CLR = np.zeros((N, rndgrid, dg))
        sim_test = np.zeros((rndgrid, dg, B_size))

        for i in range(dg):
            u_hat_CLR[:, i] = g[:, i] - np.sum(g[:, i] * temp_k, axis=1) / (1e-8 + np.sum(temp_k, axis=1))
            temp3[:, :, i] = u_hat_CLR[:, i][:, None] * kern_values_CLR
            sigma_hat_CLR[:, i] = np.sqrt(np.sum(temp3[:, :, i] ** 2, axis=0)) + 1e-8
            std_m_hat_CLR[:, i] = np.sum(g[:, i][:, None] * kern_values_CLR, axis=0) / sigma_hat_CLR[:, i]
            w_hat_CLR[:, :, i] = temp3[:, :, i] / sigma_hat_CLR[:, i]
            sim_test[:, i, :] = np.dot(w_hat_CLR[:, :, i].T, u)

        test_CLR = np.min(std_m_hat_CLR)  # The CLR test statistic

        # find critical value
        test_b_CLR = np.min(sim_test.reshape(rndgrid * dg, B_size), axis=0)

        # use moment selection to get refined critical value
        q_hat_CLR = np.quantile(test_b_CLR, tuning)
        v_hat_CLR = std_m_hat_CLR <= (-2 * q_hat_CLR)  # Construction of the estimated contact set

        test_sim_refine = []

        for i in range(dg):
            selected_indices = np.where(v_hat_CLR[:, i])[0]  # Find indices where condition is true
            if selected_indices.size > 0:
                your_matrix = np.dot(w_hat_CLR[:, selected_indices, i].T, u)
                test_sim_refine.append(your_matrix)

        if test_sim_refine:
            test_b_CLR = np.min(np.vstack(test_sim_refine), axis=0)
            q_hat_CLR = np.quantile(test_b_CLR, alpha)  # The refined CLR test critical value using the estimated contact set

        t_result = test_CLR >= q_hat_CLR  # we fail to reject the candidate parameter

        return t_result

    # Estimate the identified set
    ngrid = 50
    tgrid = np.linspace(-3, 3, ngrid)
    test_result = np.zeros(ngrid)

    for s in range(ngrid):
        theta = np.array([1, tgrid[s]])
        test_result[s] = test(theta)

    t_CI = tgrid[test_result == 1]  # obtain confidence interval

    if len(t_CI) == 0:  # determine whether CI is empty
        l_est[r] = -100
        u_est[r] = 100
    else:
        l_est[r] = np.min(t_CI)
        u_est[r] = np.max(t_CI)


## simulation ends ##

# Identified set
l_est = l_est[l_est >= -3]  # Drop empty CI
u_est = u_est[u_est <= 3]   

# evaluation metrics of the confidence interval
c = np.mean((l_est <= theta0[1]) & (theta0[1] <= u_est))    # coverage
power = np.mean(0 <= l_est)                                 # power
length = np.mean(u_est - l_est)                             # range
l_ave = np.mean(l_est)                                      # mean of lower bound
l_mad = np.mean(np.abs(l_est - theta0[1]))                  # MAD of lower bound
l_sd = np.std(l_est)                                        # SD of lower bound
u_ave = np.mean(u_est)                                      # mean of upper bound
u_mad = np.mean(np.abs(u_est - theta0[1]))                  # MAD of upper bound
u_sd = np.std(u_est)                                        # SD of upper bound

s_result = [l_ave, u_ave, c, length, power, l_mad, u_mad]   # collect evaluation metrics

print(np.round(s_result, 3))  # Print results