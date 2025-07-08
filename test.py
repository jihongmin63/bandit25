import numpy as np
import matplotlib.pyplot as plt

# Env setting

np.random.seed(42)
variance = 10
noise = 10
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
    sigma_post = np.array([noise for i in range(10)])
    sigma_0 = noise

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
tot_r = np.array(tot_r)

x = [i for i in range(horizon)]
for e in range(3) :
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title('Reward accumulation over strategies, ' + env_name[e])
    axes[1].set_title('(Relative) Reward accumulation over strategies, ' + env_name[e])
    for i in range(4) :
        data = result_r[e][i]
        axes[0].plot(x, np.mean(data, axis = 0), linestyle='-', marker='o', label=strategy[i])
        q1 = np.percentile(data, 25, axis = 0)
        q3 = np.percentile(data, 75, axis = 0)
        axes[0].fill_between(x, q1, q3, alpha = 0.2)

        relative = np.cumsum(np.ones_like(data) * mean[e], axis = 1)
        data = data - relative
        axes[1].plot(x, np.mean(data, axis = 0), linestyle='-', marker='o', label=strategy[i])
        q1 = np.percentile(data, 25, axis = 0)
        q3 = np.percentile(data, 75, axis = 0)
        axes[1].fill_between(x, q1, q3, alpha = 0.2)
    axes[0].set_xlabel("Trial")
    axes[0].set_xticks([i for i in range(horizon)])
    axes[0].set_ylabel("Accumulated Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Trial")
    axes[1].set_xticks([i for i in range(horizon)])
    axes[1].set_ylabel("Accumulated Reward")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('images/reward accumulationm, ' + env_name[e] + '.png')

for e in range(3) :
    plt.figure(figsize=(9, 5))
    plt.title('(relative) Reward over strategies, ' + env_name[e])
    data = np.mean(tot_r[e], axis=1) - mean[e] * horizon
    q1 = np.percentile(tot_r[e]- mean[e] * horizon, 25, axis=1)
    q3 = np.percentile(tot_r[e]- mean[e] * horizon, 75, axis=1) 
    err = np.vstack([data - q1, q3 - data]) 
    plt.bar([i for i in range(4)], data, yerr = err, width = 0.6, error_kw={
    'ecolor': 'gray',
    'capsize': 4,
    'capthick': 1.5,
    'linestyle': '--'
})
    plt.xlabel("Trial")
    plt.xticks([i for i in range(4)], strategy)
    for i, val in enumerate(data):
        plt.text(i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/total reward over strategies, ' + env_name[e] + '.png')


bar_width = 0.2
offsets = [-bar_width, 0, bar_width]
x = np.array([i for i in range(4)])
plt.figure(figsize=(10, 6))
for e in range(3):
    data = np.mean(switch[e], axis=1)
    q1 = np.percentile(switch[e], 25, axis=1)
    q3 = np.percentile(switch[e], 75, axis=1)
    err = np.vstack([data - q1, q3 - data])
    
    # 각 env의 bar 위치는 x + offset
    plt.bar(
        x + offsets[e], data, width=bar_width, yerr=err,
        label=env_name[e],
        error_kw={
            'ecolor': 'gray',
            'capsize': 4,
            'capthick': 1.5,
            'linestyle': '--'
        }
    )
    
    for i in range(4):
        plt.text(x[i] + offsets[e], data[i] + 0.01, f"{data[i]:.3f}",
                 ha='center', va='bottom', fontsize=9)

plt.xticks(x, strategy)
plt.xlabel("Strategy")
plt.ylabel("Switch Ratio")
plt.title("Switch over strategies by environment")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("images/grouped switch ratio.png")
plt.show()