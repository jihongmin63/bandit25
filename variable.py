import numpy as np
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass

np.random.seed(42)

@dataclass
class Env:
    env: np.ndarray         # num x structure (x horizon if restless)
    stationary : int        # 1 -> stationary, 0 -> restless
    num: int  = None        # number of env
    structure: tuple = None # structure of bandit in each env
    horizon: int = None     # number of horizon in each task
    mean: int = None        # mean of bandits & decay center (restless)
    noise: int = None       # noise of bandits (stationary)
    noise_o : int = None    # obsevational noise of bandits (restless)
    noise_d : int = None    # step noise of bandits (restless)
    lamb : int = None       # decay parameter (restless)
    name: str = None

variance = 10
noise = 10
horizon = 70
num = 100
mean = [50, 0, -50]
expected_mean = [50, 0, -50]

# stationary, non-correlated
stationary_noncor_envs = []
env_name = ['pos', 'mix', 'neg']
for i in range(3) :
    e = Env(env = np.random.normal(mean[i], noise, 10 * num).reshape(num, 10), stationary = 1, num = num, structure = (10), horizon = horizon, mean = mean[i], noise = noise, name = f"stationary_noncor_{env_name[i]}")
    stationary_noncor_envs.append(e)

# restless
lamb, noise_o, noise_d = 0.9836, 2.8, 4
restless_envs = []
for i in range(3) :
    theta = mean[i]
    envs = []

    env = [ [np.random.normal(theta, 10)] for _ in range(4)]

    for t in range(4):
        for h in range(1, 300):
            mu_post = np.random.normal(env[t][-1], noise_o)
            mu_pre = np.random.normal(lamb * mu_post + (1 - lamb) * theta, noise_d)
            env[t].append(mu_pre)

    env = np.array(env).T

    envs.append(env)

    e = Env(env = envs, stationary = 0, num = 1, structure = (4), horizon = 300, mean = theta, noise_o = noise_o, noise_d = noise_d, lamb = lamb, name = f"restless_{env_name[i]}")
    restless_envs.append(e)

# env sample images

plt.figure(figsize = (12, 5))
x = [i+1 for i in range(300)]
for t in range(4) :
    plt.plot(x, restless_envs[1].env[0].T[t], linestyle='-')
plt.xlabel("horizon")
plt.ylabel("bandit mean")
plt.title(f"{restless_envs[1].name}")
plt.savefig(f"images/sample env_{restless_envs[1].name}.png")


ex = stationary_noncor_envs[1].env[1, :]
plt.figure(figsize=(5, 2))
plt.title(f"{stationary_noncor_envs[1].name}")
plt.imshow(ex.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig(f'images/sample env_{stationary_noncor_envs[1].name}.png')


# strategies

strategy = ['generic UCB', 'generic UCB w seperation', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)', 'relative prospect UCB w seperation', 'prospect UCB w seperation', 'relative prospect UCB (inner) w seperation']
st_varaible = {
   'generic UCB' : ['beta', 'mean', 'noise', 'temp'],
   'relative prospect UCB' : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
   'prospect UCB' : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
   'relative prospect UCB (inner)' : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
   'generic UCB w seperation' : ['beta', 'beta2', 'mean', 'noise', 'temp'],
   'relative prospect UCB w seperation' : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp'],
   'prospect UCB w seperation' : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp'],
   'relative prospect UCB (inner) w seperation' : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp']
}

variables = {
    4 : ['beta', 'mean', 'noise', 'temp'],
    5 : ['beta', 'beta2', 'mean', 'noise', 'temp'],
    6 : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
    7 : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp'],
}

n_strategies = len(strategy)

recovery_range = {
    'alpha' : [0.1, 2.0],
    'beta' : [0, 10],
    'beta2' : [0, 10],
    'lambda' : [0, 10],
    'mean' : [-50, 50],
    'noise' : [0.1, 20],
    'temp' : [0.1, 5]
}

n_recovery = 1000
