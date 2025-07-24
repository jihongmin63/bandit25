import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.optimize import minimize
import itertools
import json
from tqdm import tqdm

from variable import variance, noise, horizon, num, mean, expected_mean, tot_env, env_name, strategy, variables, combinations
from function import prospect, select_arm, get_action, simulate

def cross_entropy_loss(params, actions, rewards, envs, option) :
    res = 0

    alpha, beta2, lam = 1, 1, 1
    if (len(params) == 4) :
        beta, mean, noise, temp = params[0], params[1], params[2], params[3]
    elif (len(params) == 5) :
        beta, beta2, mean, noise, temp = params[0], params[1], params[2], params[3], params[4]
    elif (len(params) == 6) :
        alpha, beta, lam, mean, noise, temp = params[0], params[1], params[2], params[3], params[4], params[5]
    else :
        alpha, beta, beta2, lam, mean, noise, temp = params[0], params[1], params[2], params[3], params[4], params[5], params[6]

    for num_e, env in enumerate(envs) :
        result = 0
        mu_post = np.array([mean for i in range(10)])
        sigma_post = np.array([noise for i in range(10)])
        sigma_0 = noise

        epsilon = 1e-9

        for t in range(horizon):
            action, reward = actions[num_e][t], rewards[num_e][t]

            p = get_action(mu_post, sigma_post, option, result, alpha = alpha, beta = beta, lam = lam)
            p = np.exp((p - np.max(p)) / temp)
            p = p / np.sum(p)

            res -= np.log(p[action] + epsilon)

            result += reward
            K_t = sigma_post[action] / (sigma_post[action] + sigma_0)
            sigma_post[action] = (1-K_t) * sigma_post[action]
            mu_post[action] += K_t * (reward - mu_post[action])

    return res    


# generate data (on mixed env)

envs = tot_env[1]
params = []

s = 0
name = strategy[s]
if (name) :
    print("["+name+"]")
    for var in tqdm(combinations[name]) :
        actions, rewards = [], []
        for env in envs :
            a, r, _, _ = simulate(var, s, env)
            actions.append(a)
            rewards.append(r)

        if(len(var) == 4) :
            initial_params = [1.0, 0, 10.0, 0.1] # 'beta', 'mean', 'noise', 'temp'
            bounds = [(0, None), (-100, 100), (0.1, 20), (0.01, 10)] 
        elif(len(var) == 5) :
            initial_params = [1.0, 1.0, 0, 10.0, 0.1] # 'beta', 'beta2', 'mean', 'noise', 'temp'
            bounds = [(0, None), (0, None), (-100, 100), (0.1, 20), (0.01, 10)] 
        elif(len(var) == 6) :
            initial_params = [1.0, 1.0, 1.0, 0, 10.0, 0.1] # 'alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'
            bounds = [(0.1, 2.0), (0, None), (0, None), (-100, 100), (0.1, 20), (0.01, 10)] 
        else :
            initial_params = [1.0, 1.0, 1.0, 1.0, 0, 10.0, 0.1] # 'alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp'
            bounds = [(0.1, 2.0), (0, None), (0, None), (0, None), (-100, 100), (0.1, 20), (0.01, 10)] 

        result = minimize(cross_entropy_loss, initial_params,
                          args = (actions, rewards, envs, s),
                          method = 'L-BFGS-B',
                          bounds = bounds
                          )

        params.append(result.x.tolist())
    
    with open("params/recovered_params.json", "r") as f:
        recovered_params = np.array(json.load(f))
        recovered_params.append(params)
    with open("params/recovered_params.json", "w") as f:
        json.dump(recovered_params, f, indent=4)