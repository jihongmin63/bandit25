import torch


def select_arm(n, env, noise) :
    """
        Args:
            n (tensor, (batch, arms)) : one-hot vectors expressing idx
            env (tensor, (batch, arms)) : means of arms
            noise (tensor, (batch,)) : observational noise

        Returns:
            (tensor, (batch,)) : the output of selected arms w/ noise

    """
    reward = (env * n).sum(dim=-1)
    return (reward + torch.randn_like(reward) * noise)


def prospect(x, loss, risk_plus, risk_minus, xi = 0, prev_result = 0, eps = 1e-12):
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
    d = x - xi * prev_result
    gain = torch.pow(torch.clamp(d, min=eps), risk_plus)
    pain = torch.pow(torch.clamp(-d, min=eps), risk_minus)
    # mask each domain, otherwise eps ** risk (not ~0 for small risk) leaks into the other side
    return gain * (d > 0) - loss * pain * (d < 0)


def vectorized_simulate(variables, envs, model, test = 0, real_actions = None, real_rewards = None) :
    """
        the function simulating agents' behavior, one agent per env

        Args:
            variables (tensor, (n_params, batch, 1)) : parameters of each agent, in the order of free_variables
            envs (list of Env) : env of each agent
            model (class defined in vectorized_models.py) : some class including calculate, update

            test (int) : 0 in general, 1 if it is for calculating log-likelihood
            real_actions (list of tensors, (batch, arms)) : one-hot arms selected by agents
            real_rewards (list of tensors, (batch, 1)) : rewards by agents

        Returns:
            action_list (list of tensors, (batch, arms)) : one-hot selected arms (flatten idx)
            reward_list (list of tensors, (batch, 1)) : the rewards of selected arms
            result (tensor, (batch, 1)) : accumulated reward
            switch (tensor, (batch,)) : switch rate over trials
            prob (tensor, (batch,)) : log-likelihood, only if test
    """
    al, rl = [], []
    result, prob = 0, 0.0

    model = model(variables, envs)
    horizon = envs[0].horizon

    if not test :
        # means of every arm at every trial, (batch, horizon, arms)
        env_means = torch.stack([env.env.flatten().expand(horizon, -1) if env.stationary else env.env.reshape(horizon, -1)
                                 for env in envs]).to(variables.device)
        env_noise = torch.tensor([env.noise for env in envs], dtype=env_means.dtype, device=variables.device)

    for t in range(horizon):
        temptation = model.calculate(t)
        probs = torch.softmax(temptation, dim=-1)

        if test :
            action = real_actions[t]
            reward = real_rewards[t]
        else :
            with torch.no_grad() :
                idx = torch.multinomial(probs, num_samples=1, replacement=True)  # (batch, 1)
                idx = idx.squeeze(-1)
                action = torch.nn.functional.one_hot(idx, num_classes=probs.size(-1)).float()
            reward = select_arm(action, env_means[:, t], env_noise).unsqueeze(-1)

        model.update(action, reward)

        al.append(action)
        rl.append(reward)
        result += reward
        if test :
            prob = prob + (action * torch.log_softmax(temptation, dim=-1)).sum(dim=-1)

    actions = torch.stack(al)  # (horizon, batch, arms)
    switch = (actions[1:] != actions[:-1]).any(dim=-1).float().mean(dim=0)

    if test :
        return al, rl, result, switch, prob
    else :
        return al, rl, result, switch
