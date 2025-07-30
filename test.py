import numpy as np
import json
import seaborn as sns
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr, sem

from variable import variance, noise, horizon, num, mean, expected_mean, test_env, env_name, strategy, variables, recovery_set, combinations, n_envs, n_strategies, st_varaible, recovery_range, n_recovery, tot_env
from function import prospect, select_arm, simulate

def behavior_data(parameter, recovered_parameter, s) :
    envs = tot_env[1]
    num_max_bandit, num_used_bandit = [], []
    recovered_num_max_bandit, recovered_num_used_bandit = [], []

    for env in envs :
        max_idx = np.argmax(env)
        actions_list, _, _, _ = simulate(parameter, s, env)
        recovered_actions_list, _, _, _ = simulate(recovered_parameter, s, env)

        num_max_bandit.append(np.sum(actions_list == max_idx))
        recovered_num_max_bandit.append(np.sum(recovered_actions_list == max_idx))

        num_used_bandit.append(len(np.unique(actions_list)))
        recovered_num_used_bandit.append(len(np.unique(recovered_actions_list)))

    return num_max_bandit, num_used_bandit, recovered_num_max_bandit, recovered_num_used_bandit

num_max_bandits = [[] for i in range(n_strategies)]
recovered_num_max_bandits = [[] for i in range(n_strategies)]

num_used_bandits = [[] for i in range(n_strategies)]
recovered_num_used_bandits = [[] for i in range(n_strategies)]

print("<Parameter recovery>")
for s, name in enumerate(strategy):
    print(f"- {name}", end="")
    try:
        with open(f"params/parameter_recovery/recovered_params_{name}_optuna.json", "r") as f:
            params, recovered_params = json.load(f)
    except FileNotFoundError:
        print(" is not recovered")
        continue
    
    for p, rp in tqdm(zip(params, recovered_params), total=len(params)):
        num_max_bandit, num_used_bandit, recovered_num_max_bandit, recovered_num_used_bandit = behavior_data(p, rp, s)
        num_max_bandits[s].append(num_max_bandit)
        recovered_num_max_bandits[s].append(recovered_num_max_bandit)

        num_used_bandits[s].append(num_used_bandit)
        recovered_num_used_bandits[s].append(recovered_num_used_bandit)

    variables = st_varaible[name]
    n_vars = len(variables)

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for k in range(n_rows * n_cols):
        ax = axes[k]
        if k < n_vars:
            key = variables[k]
            x = [params[i][k] for i in range(n_recovery)]
            y = [recovered_params[i][k] for i in range(n_recovery)]

            r, p = pearsonr(x, y)

            h = ax.hist2d(x, y, bins=50, cmap="Reds")
            fig.colorbar(h[3], ax=ax, label="Density")

            ax.plot(
                [recovery_range[key][0], recovery_range[key][1]],
                [recovery_range[key][0], recovery_range[key][1]],
                'r--',
                label='y = x'
            )

            ax.text(
                0.05, 0.95, f"r, p = {r:.2f}, {p:.2f}",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=10
            )

            ax.set_title(f"{key}", fontsize=14)
            ax.set_xlabel(f"True {key}")
            ax.set_ylabel(f"Recovered {key}")
            ax.legend(loc="lower right")
        else:
            ax.axis("off")

    fig.suptitle(f"Parameter Recovery on {name}", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f"images/recovery/{name}/parameter recovery on {name}.png")
    print("")


print("<Behavioral recovery>")
for s, name in enumerate(strategy):
    print(f"- {name}", end = "")
    try:
        with open(f"params/parameter_recovery/recovered_switchs_{name}_optuna.json", "r") as f:
            switchs, recovered_switchs = json.load(f)
    except FileNotFoundError:
        print(" is not recovered")
        continue
    
    # switch
    x = np.concatenate(switchs, axis=0)
    y = np.concatenate(recovered_switchs, axis=0)

    r, p = pearsonr(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    h = ax1.hist2d(x, y, bins=30, cmap="Reds")
    plt.colorbar(h[3], ax=ax1, label="Density")

    ax1.plot([0, 1], [0, 1], 'r--', label='y = x')
    ax1.text(
        0.05, 0.95, f"r = {r:.2f}, p = {p:f}",
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10
    )

    ax1.set_title(f"Switch ratio recovery on {name}", fontsize=14)
    ax1.set_xlabel("True switch ratio")
    ax1.set_ylabel("Recovered switch ratio")
    ax1.legend(loc="lower right")

    ax2.violinplot([x, y])
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["True", "Recovered"])
    ax2.set_title("Switch ratio distribution")
    ax2.set_ylabel("Switch ratio")

    plt.tight_layout()
    plt.savefig(f"images/recovery/{name}/behavior_switch_recovery_on_{name}.png")
    plt.close()
    
    # max bandit
    x = np.concatenate(num_max_bandits[s], axis=0)
    y = np.concatenate(recovered_num_max_bandits[s], axis=0)

    r, p = pearsonr(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    h = ax1.hist2d(x, y, bins=30, cmap="Reds")
    plt.colorbar(h[3], ax=ax1, label="Density")

    ax1.plot([0, horizon], [0, horizon], 'r--', label='y = x')
    ax1.text(
        0.05, 0.95, f"r = {r:.2f}, p = {p:f}",
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10
    )

    ax1.set_title(f"# of max bandit selection recovery on {name}", fontsize=14)
    ax1.set_xlabel("True # of max bandit selection")
    ax1.set_ylabel("Recovered # of max bandit selection")
    ax1.legend(loc="lower right")

    ax2.violinplot([x, y])
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["True", "Recovered"])
    ax2.set_title("# of max bandit selection distribution")
    ax2.set_ylabel("# of max bandit selection")

    plt.tight_layout()
    plt.savefig(f"images/recovery/{name}/behavior_maxidx_recovery_on_{name}.png")
    plt.close()

    # number of bandits
    x = np.concatenate(num_used_bandits[s], axis=0)
    y = np.concatenate(recovered_num_used_bandits[s], axis=0)

    r, p = pearsonr(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    h = ax1.hist2d(x, y, bins=30, cmap="Reds")
    plt.colorbar(h[3], ax=ax1, label="Density")

    ax1.plot([0, 10], [0, 10], 'r--', label='y = x')
    ax1.text(
        0.05, 0.95, f"r = {r:.2f}, p = {p:f}",
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10
    )

    ax1.set_title(f"# of used bandits recovery on {name}", fontsize=14)
    ax1.set_xlabel("True # of used bandits")
    ax1.set_ylabel("Recovered # of used bandits")
    ax1.legend(loc="lower right")

    ax2.violinplot([x, y])
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["True", "Recovered"])
    ax2.set_title("# of used bandits distribution")
    ax2.set_ylabel("Switch ratio")

    plt.tight_layout()
    plt.savefig(f"images/recovery/{name}/behavior_usedbandit_recovery_on_{name}.png")
    plt.close()


    print("")