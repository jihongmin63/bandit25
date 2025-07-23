import numpy as np
from variable import noise, horizon

def select_arm(n, env) : # select bandit w/ noise
    return env[n] + np.random.normal(0, noise)

def prospect(x, alpha, lam):
    if isinstance(x, (float, int)):
        if x >= 0:
            return  np.power(x, alpha)
        else:
            return  -lam * np.power(-x, alpha)
    elif isinstance(x, np.ndarray):
        util = np.empty_like(x, dtype=float)

        pos = x >= 0
        neg = x < 0

        util[pos] = np.power(x[pos], alpha)
        util[neg] = -lam * np.power(-x[neg], alpha)

        return util
    
def get_action(mean, variance, option, result, alpha=0.88, beta = 1, lam=2.25) :
    if (option == 0) : # generic UCB
        return mean + beta * variance
    elif (option == 1) : # relatvie prospect UCB
        return prospect(mean + result, alpha=alpha, lam=lam) - prospect(result, alpha=alpha, lam=lam) + beta * variance
    elif (option == 2) : # prospect UCB
        return prospect(mean, alpha=alpha, lam=lam) + beta * variance
    elif (option == 3) : # relative prospect UCB (inner)
        return prospect(mean + beta * variance + result, alpha=alpha, lam=lam) - prospect(result, alpha=alpha, lam=lam)
 
    return 0

def simulate(variables, option, env) :
    alpha, beta, lam, mean, noise, temp = variables[0], variables[1], variables[2], variables[3], variables[4], variables[5]
    
    al, rl = [], []
    result, switch, directed, random = 0, 0, 0, 0

    mu_post = np.array([mean for i in range(10)])
    sigma_post = np.array([noise for i in range(10)])
    sigma_0 = noise

    for t in range(horizon):
        values = get_action(mu_post, sigma_post, option, result, alpha = alpha, beta = beta, lam = lam)
        values = values / temp
        values = np.exp(values - np.max(values))
        values = values / np.sum(values)
        action = np.random.choice(len(values), p=values)
        reward = select_arm(action, env)

        al.append(action)
        rl.append(reward)
        result += reward

        K_t = sigma_post[action] / (sigma_post[action] + sigma_0)
        sigma_post[action] = (1-K_t) * sigma_post[action]
        mu_post[action] += K_t * (reward - mu_post[action])

    for t in range(1, horizon) :
        if(al[t-1] != al[t]) :
            switch += 1 / (horizon-1)

    return al, rl, result, switch