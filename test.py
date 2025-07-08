import numpy as np
import matplotlib.pyplot as plt

# Env setting

np.random.seed(42)
variance = 10
noise = 5
horizon = 20
num = 300
mean = [50, 0, -50]

mix_env = np.random.normal(mean[1], variance, 10 * num).reshape(num, 10)
pos_env = np.random.normal(mean[0], variance, 10 * num).reshape(num, 10)
neg_env = np.random.normal(mean[2], variance, 10 * num).reshape(num, 10)
tot_env = [pos_env, mix_env, neg_env]
env_name = ['pos', 'mix', 'neg']

def select_arm(n, env) : # select bandit w/ noise
    return env[n] + np.random.normal(0, noise)

ex = mix_env[0, :]
plt.figure(figsize=(5, 2))
plt.title("sample multi armed bandit")
plt.imshow(ex.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig('images/sample env.png')


# Agent simulation - Kalman Filter based
strategy = ['generic UCB', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)']
def simulate(option, env, mean, hori = horizon) :
    al, rl = [], []
    result, switch, directed, random = 0, 0, 0, 0

    mu_post = np.array([mean for i in range(10)])
    sigma_post = np.array([10 for i in range(10)])
    sigma_0 = 10

    for t in range(hori):
        action = get_action(mu_post, sigma_post, option, result)
        reward = select_arm(action, env)

        al.append(action)
        rl.append(reward)
        result += reward

        K_t = sigma_post[action] / (sigma_post[action] + sigma_0)
        sigma_post[action] = (1-K_t) * sigma_post[action]
        mu_post[action] += K_t * (reward - mu_post[action])

    for t in range(1, hori) :
        if(al[t-1] != al[t]) :
            switch += 1 / (hori-1)

    return al, rl, result, switch

def prospect(x, alpha=0.88, beta=0.88, lam=2.25):
    if isinstance(x, (float, int)):
        if x >= 0:
            return  np.power(x, alpha)
        else:
            return  -lam * np.power(-x, beta)
    elif isinstance(x, np.ndarray):
        util = np.empty_like(x, dtype=float)

        pos = x >= 0
        neg = x < 0

        util[pos] = np.power(x[pos], alpha)
        util[neg] = -lam * np.power(-x[neg], beta)

        return util

def get_action(mean, variance, option, result) :
    if (option == 0) : # generic UCB
        return np.argmax(mean + variance)
    elif (option == 1) : # relatvie prospect UCB
        return np.argmax(prospect(mean + result) - prospect(result) + variance)
    elif (option == 2) : # prospect UCB
        return np.argmax(prospect(mean) + variance)
    elif (option == 3) : # relative prospect UCB (inner)
        return np.argmax(prospect(mean + variance + result) - prospect(result))
 
    return 0


result_r = [[], [], []] # env * strategy * trial * horizon
tot_r = [[], [], []] # env * strategy * trial
switch = [[], [], []] # env * strategy * trial
for i in range(4):
    for e in range(3):
        result_r[e].append([])
        tot_r[e].append([])
        switch[e].append([])

    for t in range(num):
        for e in range(3):
            env_input = tot_env[e][t, :]
            _, r, total, s = simulate(i, env_input, mean[e])
            result_r[e][i].append(np.cumsum(r))
            tot_r[e][i].append(total)
            switch[e][i].append(s)

x = [i for i in range(horizon)]
for e in range(3) :
    plt.figure(figsize=(6, 5))
    plt.title('Reward accumulation over strategies, ' + env_name[e])
    for i in range(4) :
        data = result_r[e][i]
        plt.plot(x, np.mean(data, axis = 0), linestyle='-', marker='o', label=strategy[i])
        q1 = np.percentile(data, 25, axis = 0)
        q3 = np.percentile(data, 75, axis = 0)

        plt.fill_between(x, q1, q3, alpha = 0.2)
    plt.xlabel("Trial")
    plt.xticks([i for i in range(horizon)])
    plt.ylabel("Accumulated Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/reward accumulationm, ' + env_name[e] + '.png')

for e in range(3) :
    plt.figure(figsize=(9, 5))
    plt.title('Reward over strategies, ' + env_name[e])
    data = np.mean(tot_r[e], axis=1) 
    q1 = np.percentile(tot_r[e], 25, axis=1)
    q3 = np.percentile(tot_r[e], 75, axis=1) 
    err = np.vstack([data - q1, q3 - data]) 
    plt.bar([i for i in range(4)], data, yerr = err, width = 0.6, error_kw={
    'ecolor': 'gray',
    'capsize': 4,
    'capthick': 1.5,
    'linestyle': '--'
})
    plt.xlabel("Trial")
    plt.xticks([i for i in range(4)], strategy)
    for i, val in enumerate(np.mean(tot_r[e], axis = 1)):
        plt.text(i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/total reward over strategies, ' + env_name[e] + '.png')

for e in range(3) :
    plt.figure(figsize=(9, 5))
    plt.title('Switch over strategies, ' + env_name[e])
    data = np.mean(switch[e], axis=1) 
    q1 = np.percentile(switch[e], 25, axis=1)
    q3 = np.percentile(switch[e], 75, axis=1) 
    err = np.vstack([data - q1, q3 - data]) 
    plt.bar([i for i in range(4)], data, yerr = err, width = 0.6, error_kw={
    'ecolor': 'gray',
    'capsize': 4,
    'capthick': 1.5,
    'linestyle': '--'
})
    plt.xlabel("Trial")
    plt.xticks([i for i in range(4)], strategy)
    for i, val in enumerate(np.mean(switch[e], axis = 1)):
        plt.text(i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    plt.ylabel("Switch ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/switch over strategies, ' + env_name[e] + '.png')

"""
switch = [0 for i in range(4)]
for s in range(4) :
    for i in range(1, horizon) :
        if (a[s][i-1] != a[s][i]) :
            switch[s] += 1
    switch[s] /= (horizon-1)

plt.figure(figsize=(10, 5))
plt.title('# of Switches over strategies')
plt.bar([i for i in range(4)], switch, width = 0.6)
plt.xticks([i for i in range(4)], strategy)
for i, val in enumerate(switch):
    plt.text(i, val - 0.05, f"{val:.3f}", ha='center', va='bottom', fontsize=10, color="white")
plt.savefig('switch rate.png')
"""