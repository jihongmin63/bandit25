import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr

from variable import variance, noise, horizon, num, mean, expected_mean, test_env, env_name, strategy, variables, init, recovery_set, combination
from function import prospect, select_arm, simulate

with open("recovered_params.json", "r") as f:
    recovered_params = np.array(json.load(f))
with open("simulated_rewards_"+name+".json", "r") as f:
    tot_rewards = np.array(json.load(f))
with open("simulated_switchs_"+name+".json", "r") as f:
    tot_switchs = np.array(json.load(f)) # strategy x env x var

for n, name in enumerate(strategy):
    for k, key in enumerate(recovery_set.keys()):
        x = [combination[i][k] for i in range(len(combination))]
        y = [recovered_params[n][i][k] for i in range(len(combination))]

        r, p = pearsonr(x, y)

        plt.figure(figsize=(6, 5))
        plt.title(f"Parameter Recovery of {key} on {name}")
        plt.scatter(x, y, alpha=0.1, s=2, label='Recovered')

        min_val, max_val = min(x), max(x)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        plt.text(0.05, 0.95, f"r = {r:.2f}", transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='left', fontsize=10)

        plt.xlabel(f"True {key}")
        plt.ylabel(f"Recovered {key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"images/parameter recovery of {key} on {name}.png")
        plt.close()


# 필요한 것
# mu 전체 기반 lambda vs switch rate over envs
# 특정 mu에 대해 반복
# mu x lambda x env x sws

mu_values = recovered_params['mu']
lambda_values = recovered_params['lambda']
n_envs = 3
n_strategies = len(strategies)

# 인덱스: [strategy][mu][lambda][env] → list of switch rates
switchs_arranged = [[[[[] for _ in range(n_envs)]
                      for _ in range(len(lambda_values))]
                     for _ in range(len(mu_values))]
                    for _ in range(n_strategies))]

# 미리 인덱스: [mu][lambda] → combination index list
indices = [[[i for i, c in enumerate(combination)
             if c[3] == mu and c[2] == lamb]
            for lamb in lambda_values]
           for mu in mu_values]

# 데이터 채우기
for s in range(n_strategies):
    for e in range(n_envs):
        for i, mu in enumerate(mu_values):
            for j, lamb in enumerate(lambda_values):
                for idx in indices[i][j]:
                    switchs_arranged[s][i][j][e].append(tot_switchs[s][e][idx])

for s, strat_name in enumerate(strategies):
    fig, axes = plt.subplots(1, len(mu_values), figsize=(4 * len(mu_values), 4), sharey=True)
    axes = axes.flatten()

    for i, mu in enumerate(mu_values):
        ax = axes[i]
        for e in range(n_envs):
            means = []
            for j, lamb in enumerate(lambda_values):
                vals = switchs_arranged[s][i][j][e]
                mean_val = np.mean(vals) if len(vals) > 0 else np.nan
                means.append(mean_val)
            ax.plot(lambda_values, means, marker='o', label=f'env {e}')

        ax.set_title(f'{strat_name}, mu={mu}')
        ax.set_xlabel('Lambda')
        if i == 0:
            ax.set_ylabel('Switch Rate')
        ax.grid(True)
        ax.legend()

    plt.suptitle(f'Strategy: {strat_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"Switch rate over lambda, mu, env on {strat_name}")


"""
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