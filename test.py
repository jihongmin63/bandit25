import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from variable import variance, noise, horizon, num, mean, expected_mean, test_env, env_name, strategy, variables, init
from function import prospect, select_arm, get_action, simulate

with open("best_params.json", "r") as f:
    best_params = json.load(f)

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
            variable = []
            dic = best_params[strategy[i]]

            for n in variables :
                if n in dic :
                    variable.append(dic[n])
                else :
                    variable.append(init[n])

            variable[3] = mean[e]
            env_input = test_env[e][t, :]
            _, r, total, s = simulate(variable, i, env_input)
            result_r[e][i].append(np.cumsum(r))
            tot_r[e][i].append(total)
            switch[e][i].append(s)

tot_r = np.array(tot_r)


# Results

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
    axes[0].set_xticks([i*10 for i in range(int(horizon/10))])
    axes[0].set_ylabel("Accumulated Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Trial")
    axes[1].set_xticks([i*10 for i in range(int(horizon/10))])
    axes[1].set_ylabel("Accumulated Reward")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('images/reward accumulationm, ' + env_name[e] + '.png')

for e in range(3) :
    plt.figure(figsize=(9, 5))
    plt.title('(relative) Reward over strategies, ' + env_name[e])
    data = np.mean(tot_r[e], axis=1) - mean[e] * horizon
    q1 = np.percentile(tot_r[e] - mean[e] * horizon, 25, axis=1)
    q3 = np.percentile(tot_r[e] - mean[e] * horizon, 75, axis=1) 
    err = np.vstack([data - q1, q3 - data]) 
    plt.bar([i for i in range(4)], data, yerr = err, width = 0.6, error_kw={
    'ecolor': 'gray',
    'capsize': 4,
    'capthick': 1.5,
    'linestyle': '--'
})
    plt.xlabel("Trial")
    plt.xticks([i for i in range(4)], strategy)
    plt.yticks([100 * i for i in range(1, 6)])
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
    
    plt.bar(
        x + offsets[e], data, width=bar_width, yerr=err,
        label=env_name[e],
        error_kw={'ecolor': 'gray', 'capsize': 4, 'capthick': 1.5, 'linestyle': '--'}
    )
    
    for i in range(4):
        plt.text(x[i] + offsets[e], data[i] + 0.01, f"{data[i]:.3f}",
                 ha='center', va='bottom', fontsize=9)


pairs = list(itertools.combinations(range(3), 2))

for i in range(4):
    base = np.max([np.mean(switch[e][i]) + np.std(switch[e][i]) for e in range(3)]) + 0.02
    
    for j, (a, b) in enumerate(pairs):
        stat, pval = ttest_ind(switch[a][i], switch[b][i])
        
        if pval < 0.001:
            label = '***'
        elif pval < 0.01:
            label = '**'
        elif pval < 0.05:
            label = '*'
        else:
            continue

        h = base + 0.03 * j
        x1 = x[i] + offsets[a]
        x2 = x[i] + offsets[b]
        xm = (x1 + x2) / 2

        plt.plot([x1, x1, x2, x2], [h, h+0.003, h+0.003, h], lw=1.2, c='black')
        plt.text(xm, h + 0.005, label, ha='center', va='bottom', fontsize=10)

plt.xticks(x, strategy)
plt.xlabel("Strategy")
plt.ylabel("Switch Ratio")
plt.title("Switch over strategies by environment")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("images/grouped_switch_ratio_with_pvalues.png")