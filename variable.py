import numpy as np
import matplotlib.pyplot as plt
import itertools

variance = 10
noise = 10
horizon = 30
num = 100
mean = [50, 0, -50]
expected_mean = [50, 0, -50]


np.random.seed(42)
mix_env = np.random.normal(mean[1], variance, 10 * num).reshape(num, 10)
pos_env = np.random.normal(mean[0], variance, 10 * num).reshape(num, 10)
neg_env = np.random.normal(mean[2], variance, 10 * num).reshape(num, 10)
tot_env = [pos_env, mix_env, neg_env]
test_env = [np.random.normal(mean[i], variance, 10 * num).reshape(num, 10) for i in range(3)]
env_name = ['pos', 'mix', 'neg']

strategy = ['generic UCB', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)', 'generic UCB w/ seperation', 'relative prospect UCB w/ seperation', 'prospect UCB w/ seperation', 'relative prospect UCB (inner) w/ seperation']
st_varaible = {
   'generic UCB' : ['beta', 'mean', 'noise', 'temp'],
   'relative prospect UCB' : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
   'prospect UCB' : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
   'relative prospect UCB (inner)' : ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp'],
   'generic UCB w/ seperation' : ['beta', 'beta2', 'mean', 'noise', 'temp'],
   'relative prospect UCB w/ seperation' : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp'],
   'prospect UCB w/ seperation' : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp'],
   'relative prospect UCB (inner) w/ seperation' : ['alpha', 'beta', 'beta2', 'lambda', 'mean', 'noise', 'temp']
}

variables = ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp']

n_envs = len(tot_env)
n_strategies = len(strategy)

recovery_set = {
    'alpha' : [0.4, 0.6, 0.8, 1.0, 1.5],
    'beta' : [0, 1, 2, 5, 7, 10],
    'beta2' : [0, 1, 2, 5, 7, 10],
    'lambda' : [0.5, 1.0, 2.0, 3.0],
    'mean' : [-50, -25, -10, 0, 10, 25, 50],
    'noise' : [1, 2, 5, 10, 15],
    'temp': [0.1, 0.5, 1.0, 5.0],
}

combinations = {}
for name in strategy :
    params_set = {}
    for param in st_varaible[name] :
        params_set[param] = recovery_set[param]
    combinations[name] = list(itertools.product(*list(params_set.values())))

ex = mix_env[0, :]
plt.figure(figsize=(5, 2))
plt.title("sample multi armed bandit")
plt.imshow(ex.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig('images/sample env.png')
