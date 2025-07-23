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

strategy = ['generic UCB', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)']

variables = ['alpha', 'beta', 'lambda', 'mean', 'noise', 'temp']

ex = mix_env[0, :]
plt.figure(figsize=(5, 2))
plt.title("sample multi armed bandit")
plt.imshow(ex.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig('images/sample env.png')

recovery_set = {
    'alpha' : [0.4, 0.6, 0.8, 1.0, 1.5],
    'beta' : [0, 1, 2, 5, 7, 10],
    'lambda' : [0.5, 1.0, 2.0, 3.0],
    'mu' : [-50, -25, -10, 0, 10, 25, 50],
    'sigma' : [1, 2, 5, 10, 15],
    'temperature': [0.1, 0.5, 1.0, 5.0]
}

combination = list(itertools.product(*list(recovery_set.values())))
#print(len(combination))