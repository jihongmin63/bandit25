import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr, sem

from variable import variance, noise, horizon, num, mean, expected_mean, test_env, env_name, strategy, variables, recovery_set, combinations, n_envs, n_strategies, st_varaible
from function import prospect, select_arm, simulate


for n, name in enumerate(strategy):
    try:
        with open(f"params/recovered_params_{name}_optuna.json", "r") as f:
            recovered_params = np.array(json.load(f))
    except FileNotFoundError:
        print(f"{name} is not recovered")
        continue
    
    for k, key in enumerate(st_varaible[name]):
        x = [combinations[name][i][k] for i in range(len(combinations[name]))]
        y = [recovered_params[i][k] for i in range(len(combinations[name]))]

        r, p = pearsonr(x, y)

        plt.figure(figsize=(6, 5))
        plt.title(f"Parameter Recovery of {key} on {name}")

        # Heatmap with white-red colormap
        plt.hist2d(x, y, bins=50, cmap="Reds")
        plt.colorbar(label="Density")

        # Diagonal line y = x
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        # Correlation coefficient text
        plt.text(0.05, 0.95, f"r = {r:.2f}", transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='left', fontsize=10)

        plt.xlabel(f"True {key}")
        plt.ylabel(f"Recovered {key}")

        # Fix legend position
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(f"images/recovery/parameter recovery of {key} on {name}.png")
        plt.close()

print("hello")
with open("params/simulated_switchs_over_strategies.json", "r") as f:
    tot_switchs = json.load(f) # strategy x env x combinations

s, name = 6, strategy[6]

# 각 환경에 대해 평균과 SEM 계산
means = []
yerr_lower = []
yerr_upper = []

for i in range(n_envs):
    vals = [np.mean(rep) for rep in tot_switchs[s][i]]
    m = np.mean(vals)
    e = sem(vals)
    means.append(m)
    yerr_lower.append(e)   # 또는 다른 값
    yerr_upper.append(e)   # 또는 다르게 조정

# 2xN 배열로 구성
yerr = np.array([yerr_lower, yerr_upper])

# 시각화
plt.figure(figsize=(6, 5))
plt.bar(range(n_envs), means, yerr=yerr, capsize=5)
plt.xticks(range(n_envs), [env_name[i] for i in range(n_envs)])
plt.ylabel("Switch Ratio")
plt.tight_layout()
plt.savefig("prospect UCB w separation exploration (1, 10).png")


"""
indices = [[[
    [i for i, c in enumerate(combinations[name]) if c[3] == mu and c[2] == lamb]
    for lamb in st_varaible[name]['lambda']]
    for mu in st_varaible[name]['mean']]
    for s, name in enumerate(strategy)
]

switchs_arranged = [[[
    [np.concatenate([tot_switchs[s][e][k] for k in indices[s][i][j]])
     for e in range(n_envs)]
    for j in range(len(st_varaible[name]['lambda']))]
    for i in range(len(st_varaible[name]['mean']))]
    for s in range(n_strategies)]


for s, strat_name in enumerate(strategy):
    plt.figure(figsize=(6, 5))

    for e in range(n_envs):
        means = []
        lower_err = []
        upper_err = []

        if e == 0:
            i = 6
        elif e == 1:
            i = 3
        elif e == 2:
            i = 0

        for j, lamb in enumerate(lambda_values):
            vals = switchs_arranged[s][i][j][e]

            if len(vals) > 0:
                mean_val = np.mean(vals)
                q1 = np.percentile(vals, 25)
                q3 = np.percentile(vals, 75)

                means.append(mean_val)
                lower_err.append(mean_val - q1)
                upper_err.append(q3 - mean_val)

        yerr = np.array([lower_err, upper_err])
        plt.errorbar(lambda_values, means, yerr=yerr, marker='o', capsize=4,
                     label=f"{env_name[e]} on mu = {recovery_set['mean'][i]}")

    plt.xlabel('Lambda')
    plt.ylabel('Switch rate')
    plt.grid(True)
    plt.legend()
    plt.suptitle(f'Strategy: {strat_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"images/simulation/Selected switch rate simulation on {strat_name}")
    plt.close()


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
    
    for i, val in enumerate(data):
        plt.text(i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=10)

    offset = 15
    current_height = max(data) + 10

    for count, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        t_stat, p_val = ttest_ind(tot_r[e][i], tot_r[e][j], equal_var=False)
        #print(f"Env: {env_name[e]}, Comparing {strategy[i]} vs {strategy[j]} -> p = {p_val:.4f}")
        if p_val < 0.05:
            x1, x2 = i, j
            y = current_height + offset * count
            plt.plot([x1, x1, x2, x2], [y - 2, y, y, y - 2], lw=1.5, c='k')  # 수평선
            plt.text((x1 + x2) / 2, y, "*", ha='center', va='bottom', color='red', fontsize=14)

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
"""