import numpy as np
import matplotlib.pyplot as plt

variance = 10
noise = 10
horizon = 70
num = 300
mean = [50, 0, -50]
expected_mean = [50, 0, -50]

np.random.seed(42)
mix_env = np.random.normal(mean[1], variance, 10 * num).reshape(num, 10)
pos_env = mix_env + 50
neg_env = mix_env - 50
tot_env = [pos_env, mix_env, neg_env]
env_name = ['pos', 'mix', 'neg']

strategy = ['generic UCB', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)']

variables = ['alpha', 'beta', 'lambda', 'mean', 'noise']
init = {'alpha' : 0.88, 'beta' : 1, 'lambda' : 2.25, 'mean' : 0, 'noise' : 10}
variable_init = {'generic UCB' : {
        'beta' : [-10, 10], 'noise' : [0.1, 20.0] 
    },
    'relative prospect UCB' : { 
        'alpha' : [0.1, 3.0], 'beta' : [-10, 10], 'lambda' : [1.0, 10.0], 'noise' : [0.1, 20.0]   
    },
    'prospect UCB' : { 
        'alpha' : [0.1, 3.0], 'beta' : [-10, 10], 'lambda' : [1.0, 10.0], 'noise' : [0.1, 20.0]   
    },
    'relative prospect UCB (inner)' : { 
        'alpha' : [0.1, 3.0], 'beta' : [-10, 10], 'lambda' : [1.0, 10.0], 'noise' : [0.1, 20.0]   
    }
}

ex = mix_env[0, :]
plt.figure(figsize=(5, 2))
plt.title("sample multi armed bandit")
plt.imshow(ex.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig('images/sample env.png')