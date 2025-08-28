import torch
from variable import horizon, Env
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import mvn
        

def select_arm(n, env, noise) :
    """
        Args:
            n (int) : the index of arm
            env (float array) : means of arms
            noise (float) : observational noise

        Returns:
            (float) : the output of selected arm w/ noise

    """
    return env[n] + torch.normal(0, noise)

def prospect(x, loss, risk_plus, risk_minus, xi = 0, prev_result = None):
    """
        It is utility function that modify the original value

        Args:
            x (int or array) : the value that should be modified
            loss (float) : loss averse parameter
            risk_plus (float) : risk averse parameter of plus domain
            risk_minus (float) : risk averse parameter of minus domain
        
            xi (float, 0 ~ 1) : parameter dictating whether it is relative prospect function
            prev_result (float) : previous reward for relative prospect function
        Returns:
            util (int or array) : the modified value 
    """
    if xi :
        x -= xi * prev_result
    return torch.pow(torch.maximum(x, 0), risk_plus) - loss * torch.pow(torch.maximum(-x, 0), risk_minus)

# loss, risk, mean, noise, theta1, theta2, kappa, rho, beta1, beta2, phi2, phi3

def simulate(variables, env, model, test = 0, real_actions = None, real_rewards = None) :
    """
        the function simulating agent's behavior on given strategy and environment

        Args:
            varaibles (float array) : variables required for the given strategy
            option (int) : index of strategy
            env (numpy array) : mean of bandits
            model (class defined in models.py) : some function including setting, calculate, update

            test (int) : 0 in general, 1 if it is for calculating cross entropy loss
            real_actions (int array) : selected idx by agent
            real_rewards (float array) : rewards by agent
        
        Returns:
            action_list (int array) : the indexes of selected bandits (it is flatten idx)
            reward_list (float array) : the rewards of selected bandits
            result (float) : accumulated reward
            switch (float) : switch rate over trial
            prob (float) : log-likelihood
    """

    al, rl = [], []
    result, switch, directed, random = 0, 0, 0, 0
    prob, epsilon = 0, 1e-12

    model = model(variables, env)
    #model.set()

    for t in range(env.horizon):
        temptation = model.calculate(t)
        values = torch.exp(temptation - torch.max(temptation))
        values = values / torch.sum(values)

        if not test :
            if (torch.isnan(values).any()) :
                print(temptation)
            action = torch.multinomial(values, num_samples=1)
            if env.stationary :
                reward = select_arm(action, env.env.flatten(), env.noise)
            else :
                reward = select_arm(action, env.env[t].flatten(), env.noise)
            
            action = torch.unravel_index(action, env.structure)
        else :
            action = real_actions[t]
            reward = real_rewards[t]

        model.update(action, reward)
        
        al.append(action)
        rl.append(reward)
        result += reward
        prob += torch.log(values[action] + epsilon)

    for t in range(1, horizon) :
        if(al[t-1] != al[t]) :
            switch += 1 / (horizon-1)

    return al, rl, result, switch, prob

"""
    
def get_action(mean, variance, option, result, risk=0.88, beta=1, beta2=1, loss=2.25) :
    
        the function calculating values according to different strategies

        Args:
            mean (numpy array) : estimated mean for bandits
            variance (numpy array) : estimated variance for bandits
            option (int) : index for strategy
            result (int) : accmulated reward

            risk, beta, beta2, loss (int) : parameters required for specific strategy

        Returns:
            (numpy array) : calculated value
    

    if (option == 0) : # generic UCB
        return mean + beta * variance
    elif (option == 2) : # relatvie prospect UCB
        return prospect(mean + result, risk=risk, loss=loss) - prospect(result, risk=risk, loss=loss) + beta * variance
    elif (option == 3) : # prospect UCB
        return prospect(mean, risk=risk, loss=loss) + beta * variance
    elif (option == 4) : # relative prospect UCB (inner)
        return prospect(mean + beta * variance + result, riska=risk, loss=loss) - prospect(result, risk=risk, loss=loss)
    else:
        sigma = np.where(mean > 0, beta * variance / 2, beta2 * variance / 2) + np.where(mean < 0, beta2 * variance / 2, beta * variance / 2)
        if (option == 1) : # seperation of sigma
            return mean + sigma
        elif (option == 5) :
            return prospect(mean + result, risk=risk, loss=loss) - prospect(result, risk=risk, loss=loss) + sigma
        elif (option == 6) :
            return prospect(mean, risk=risk, loss=loss) + sigma
        elif (option == 7) :
            return prospect(mean + sigma + result, risk=risk, loss=loss) - prospect(result, risk=risk, loss=loss)             
    return 0
def simulate(variables, option, env, test = 0, real_actions = None, real_rewards = None) :
    
        the function simulating agent's behavior on given strategy and environment

        Args:
            varaibles (float array) : variables required for the given strategy
            option (int) : index of strategy
            env (numpy array) : mean of bandits

            test (int) : 0 in general, 1 if it is for calculating cross entropy loss
            real_actions (int array) : selected idx by agent
            real_rewards (float array) : rewards by agent
        
        Returns:
            action_list (int array) : the indexes of selected bandits (it is flatten idx)
            reward_list (float array) : the rewards of selected bandits
            result (float) : accumulated reward
            switch (float) : switch rate over trial
            prob (float) : log-likelihood
    
    if not env.stationary :
        lamb, theta, noise_o, noise_d = env.lamb, env.theta, env.noise, env.noise_d

    risk, beta, beta2, loss = 1, 1, 1, 1
    if len(variables) == 4 :
        beta, mean, noise, temp = variables
    elif len(variables) == 5:
        beta, beta2, mean, noise, temp = variables
    elif len(variables) == 6:
        risk, beta, loss, mean, noise, temp = variables
    elif len(variables) == 7:
        risk, beta, beta2, loss, mean, noise, temp = variables
    
    al, rl = [], []
    result, switch, directed, random = 0, 0, 0, 0
    prob, epsilon = 0, 1e-12

    mu_pre = np.ones(env.structure, dtype = float) * mean
    sigma_pre = np.ones(env.structure, dtype = float) * noise

    for t in range(env.horizon):
        values = get_action(mu_pre, sigma_pre, option, result, risk=risk, beta=beta, beta2=beta2, loss=loss)
        values = values / temp
        values = np.exp(values - np.max(values))
        values = values / np.sum(values)
        values = values.flatten()

        if not test :
            action = np.random.choice(len(values), p=values)
            if env.stationary :
                reward = select_arm(action, env.env.flatten(), env.noise)
            else :
                reward = select_arm(action, env.env[t].flatten(), env.noise)
        else :
            action = real_actions[t]
            reward = real_rewards[t]
        
        al.append(action)
        rl.append(reward)
        result += reward
        prob += np.log(values[action] + epsilon)

        K_t = (np.square(sigma_pre[action])) / (np.square(sigma_pre[action]) + noise_o ** 2)
        
        mu_post = mu_pre
        mu_post[action] += (reward - mu_pre[action]) * K_t

        sigma_post = sigma_pre
        sigma_post[action] = np.sqrt((1 - K_t) * np.square(sigma_pre[action]))

        if env.stationary :
            mu_pre = mu_post
            sigma_pre = sigma_post
        else :
            mu_pre = lamb * mu_post + (1-lamb) * theta
            sigma_pre = np.sqrt(np.square(lamb) * np.square(sigma_post) + np.square(noise_d))

    for t in range(1, horizon) :
        if(al[t-1] != al[t]) :
            switch += 1 / (horizon-1)

    return al, rl, result, switch, prob
"""