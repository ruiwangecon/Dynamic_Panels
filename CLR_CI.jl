# this code is for the project 'Identification of Nonlinear Dynamic Panels under Partial Stationarity', by Wayne Gao and Rui Wang
# it is implemneting CLR inference method to compute confidence interval for the static ordered choice model

using Random, LinearAlgebra, Distributions
using StableRNGs: StableRNG


# Set the seed for reproducibility
rng = StableRNG(123)

# Initialize simulation configurations
alpha = 0.05             # Significance level
T = 2                    # Time period
theta0 = [1; 1]          # True parameter
N = 2000                 # Sample size

sigma_z = 1.0            # Standard deviation of Z: {1, 1.5, 2}
mu = [0, 0]              # Mean
rho = 0.25               # Correlation between two periods: {0, 0.25, 0.5}
sigma = [1 rho; rho 1]   # Variance matrix

R = 1000                 # Simulation repetitions
l_est = zeros(R)         # Lower bound
u_est = zeros(R)         # Upper bound
B_size = 1000            # Simulation draws for calculating the CLR test critical value
rndgrid = 1000           # The number of randomly drawn grid points
tuning = 0.1 / log(N)    # Tuning parameter for improving the CLR test power

u = randn(rng, N, B_size)     # Random multipliers used in the CLR test implementation
test_b = zeros(B_size)        # Simulated version of the test statistic
b_c = 2.65                    # Bandwidth scale

# Define kernel function
function kern(v)
    return ((15/16) * ((1 - v^2)^2)) * (abs(v) <= 1)
end

v_hat_CLR = zeros(rndgrid)   # v_hat stores the indices of v in the estimated contact set
wsize_CLR = b_c / (N^(1/9))  # Undersmoothing for d=4


## Main simulation repetitions ##

for r in 1 : R

    # DGP setup
    # Generate the true covariate 
    Z1 = rand(rng, Normal(0, sigma_z), N, T) 
    Z2 = rand(rng, Normal(0, sigma_z), N, T)
    Z = [Z1 Z2]

    # generate grid for covariates
    Zgrid1 = rand(rng, Normal(0, sigma_z), rndgrid, T)
    Zgrid2 = rand(rng, Normal(0, sigma_z), rndgrid, T)
    Zgrid = [Zgrid1 Zgrid2]

    # Generate error term
    a = (Z1 + Z2) * theta0 / (4 * T * sigma_z)   # fixed effect
    ep = rand(rng, MvNormal(mu, sigma), N)'      # time changing error

    # dependent variable
    b2 = -1       # threshold of ordered choice
    b3 = 1
    Ystar = [ep[:, 1] + a + Z1 * theta0 ep[:, 2] + a + Z2 * theta0]           # latent dependent variable 
    Y = 1 .* (Ystar .<= b2) + 2 .* (b2 .< Ystar .<= b3) + 3 .* (Ystar .> b3)  # observed dependent variable

    # define condiional moment function
    function cmom(theta)
        covind = [Z1 * theta Z2 * theta]  # covariate index
        # define four moment functions
        g1 = zeros(N, T, T)
        g2 = zeros(N, T, T)
        g3 = zeros(N, T, T)
        g4 = zeros(N, T, T)
        for s = 1 : T
            for t = 1 : T
                if s != t
                   g1[:, s, t] = (covind[:, t] - covind[:, s] .>= 0).* ((Y[:,s] .== 1) - (Y[:,t] .== 1))
                   g2[:, s, t] = (b2 .- covind[:, s] .>= b3 .- covind[:, t]).* ((Y[:,s] .== 1) - (Y[:,t] .<= 2))
                   g3[:, s, t] = (b3 .- covind[:, s] .>= b2 .- covind[:, t]).* ((Y[:,s] .<= 2) - (Y[:,t] .== 1))
                   g4[:, s, t] = (covind[:, t] - covind[:, s] .>= 0).* ((Y[:,s] .<= 2) - (Y[:,t] .<= 2))
               end
            end
        end

        g = [g1[:, 1, 2] g1[:, 2, 1] g2[:, 1, 2] g2[:, 2, 1] g3[:, 1, 2] g3[:, 2, 1] g4[:, 1, 2] g4[:, 2, 1]]
        return g
    end

    # calculate kernel values
    hsize_CLR = wsize_CLR * std(Z, dims=1)  # bandwidth
    kern_values_CLR = ones(N, rndgrid)      # initial kernel values at grid
    temp_k = ones(N, N)

    for k2 in 1 : size(Z, 2)
        kern_values_CLR .*= kern.((Zgrid[:, k2]' .- Z[:, k2]) ./ hsize_CLR[k2])
        temp_k .*= kern.((Z[:, k2]' .- Z[:, k2]) ./ hsize_CLR[k2])
    end

    # compute test statistic for a candidate parameter
    function test(theta)

        g = cmom(theta)      # conditional moment
        dg = size(g, 2)      # dimension of moments

        u_hat_CLR = zeros(N, dg)              # residual of moment
        temp3 = zeros(N, rndgrid, dg)         # uhat*kern
        sigma_hat_CLR = zeros(rndgrid, dg)    # standard deviation
        std_m_hat_CLR = zeros(rndgrid, dg)    # t statistic
        w_hat_CLR = zeros(N, rndgrid, dg)
        sim_test = zeros(rndgrid, dg, B_size) 


        for i = 1 : dg
            u_hat_CLR[:, i] = g[:, i] - sum(g[:, i] .* temp_k, dims=1)' ./ (1e-8 .+ sum(temp_k, dims=1))'
            temp3[:, :, i] = u_hat_CLR[:, i].* kern_values_CLR
            sigma_hat_CLR[:, i] = sqrt.(sum(temp3[:,:, i] .* temp3[:,:, i], dims=1))' .+ 1e-8
            std_m_hat_CLR[:, i]= sum(g[:, i] .* kern_values_CLR, dims=1)' ./ sigma_hat_CLR[:, i]
            w_hat_CLR[:, :, i] = temp3[:, :, i] ./ sigma_hat_CLR[:, i]'
            sim_test[:, i, :] = w_hat_CLR[:, :, i]' * u
        end

        test_CLR = minimum(std_m_hat_CLR)      # The CLR test statistic

        # find critical value
        test_b_CLR = minimum(reshape(sim_test, rndgrid * dg, B_size), dims=1)

        # use moment selection to get refined critical value
        q_hat_CLR = quantile(vec(test_b_CLR), tuning)
        v_hat_CLR = (std_m_hat_CLR .<= (-2 * q_hat_CLR))  # Construction of the estimated contact set

        test_sim_refine = Vector{Matrix{Float64}}()

        for i in 1:dg
            your_matrix = w_hat_CLR[:, v_hat_CLR[:, i] .== 1, i]' * u
            push!(test_sim_refine, your_matrix)
        end

        test_b_CLR = minimum(vcat(test_sim_refine...), dims=1)
        q_hat_CLR = quantile(vec(test_b_CLR), alpha)  # The refined CLR test critical value using the estimated contact set

        t_result = (test_CLR >= q_hat_CLR)       # we fail to reject the candidate parameter

        return t_result
    end

    # estimate the identified set
    ngrid = 50
    tgrid = LinRange(-1, 3, ngrid)
    test_result = zeros(ngrid)

    for s = 1 : ngrid
        theta = [1; tgrid[s]]
        test_result[s] = test(theta)
    end

    t_CI = tgrid[test_result .== 1]     # obtain confidence interval

    if isempty(t_CI)                    # determine whether CI is empty
       l_est[r] = -100
       u_est[r] = 100
    else
       l_est[r] = minimum(t_CI)            
       u_est[r] = maximum(t_CI)
    end

end

# evaluation metrics of the confidence interval
l_est = l_est[l_est .>= -1]      # drop empty CI
u_est = u_est[u_est .<= 3] 

c = mean((l_est .<= theta0[2] .<= u_est))  # coverage 
power = mean((0 .<= l_est))                # power
len = mean(u_est - l_est)                  # range 
l_ave = mean(l_est)                        # mean of lower bound 
l_mad = mean(abs.(l_est .- theta0[2]))     # MAD of lower bound 
l_sd = std(l_est)                          # SD of lower bound
u_ave = mean(u_est)                        # mean of upper bound
u_mad = mean(abs.(u_est .- theta0[2]))     # MAD od upper bound
u_sd = std(u_est)                          # SD of upper bound

s_result = [l_ave u_ave c len power l_mad u_mad]    # collect evaluation metrics

println(round.(s_result, digits=3))

