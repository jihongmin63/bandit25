import torch
from bandit25.variable import Env, mean, env_name, noise, horizon

# restless
lamb, noise_o, noise_d = 0.9836, 4, 2.8

# Draws are made in the same order as the original envs.py (fb038da), so
# torch.manual_seed(42); stationary(); restless("original") reproduces the
# environments used before c165b65. scripts/make_envs.py seeds 42 the same way.

# stationary, non-correlated
def stationary() :
    stationary_noncor_envs = [[] for _ in range(3)]
    for i in range(3) :
        for _ in range(100) :
            e = Env(env = torch.normal(mean[i], noise, size=(10,)), stationary = 1, structure = (10,), horizon = horizon, mean = mean[i], noise = noise, name = f"stationary_noncor_{env_name[i]}")
            stationary_noncor_envs[i].append(e)
    return stationary_noncor_envs


def restless(process = "daw") :
    """
        Args:
            process (str) : how the bandit means drift
                'daw' : the process the Kalman model assumes (Daw et al., 2006), step SD noise_d = 2.8
                    mu_pre' ~ N(lamb * mu_pre + (1 - lamb) * theta, noise_d)
                'original' : the environments used before; the drift also takes the observation noise (step SD ~4.8)
                    mu_post ~ N(mu_pre, noise_o),  mu_pre' ~ N(lamb * mu_post + (1 - lamb) * theta, noise_d)

        Returns:
            (list) : restless_envs[i] holds 100 envs whose means drift around mean[i]; env.env is (300, 4)
    """
    restless_envs = [[] for _ in range(3)]
    for i in range(3) :
        for _ in range(100) :
            theta = float(mean[i])
            env = [[torch.normal(theta, 10.0, size=(1,))] for _ in range(4)]

            for t in range(4):
                for h in range(1, 300):
                    mu = env[t][-1]
                    if process == "original" :
                        mu = torch.normal(mu, noise_o)
                    mu_pre = torch.normal(lamb * mu + (1 - lamb) * theta, noise_d)
                    env[t].append(mu_pre)

            env = torch.transpose(torch.Tensor(env), 0, 1)

            e = Env(env = env, stationary = 0, structure = (4,), horizon = 300, mean = theta, noise = noise_o, noise_d = noise_d, lamb = lamb, name = f"restless_{env_name[i]}")
            restless_envs[i].append(e)
    return restless_envs
