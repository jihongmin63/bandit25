import numpy as np
import json
import multiprocessing as mp
from tqdm import tqdm

from variable import variance, noise, horizon, num, mean, expected_mean, tot_env, env_name, strategy, variables, recovery_set, combinations, n_envs
from function import prospect, select_arm, get_action, simulate

def simulate_single_case(args):
    s, var, i = args
    r, sw = [], []
    for j in range(num):
        _, _, r_element, sw_element = simulate(var, s, tot_env[i][j, :])
        r.append(r_element)
        sw.append(sw_element)
    return (i, r, sw)

if __name__ == "__main__":
    rewards = [[[] for _ in range(3)] for _ in range(len(strategy))]
    switchs = [[[] for _ in range(3)] for _ in range(len(strategy))]

    for s, name in enumerate(strategy):
        print(f"[{name}]")

        task_list = []
        for var in combinations[name]:
            for i in range(n_envs):
                task_list.append((s, var, i))

        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(
                pool.imap_unordered(simulate_single_case, task_list),
                total=len(task_list),
                desc=f"{name} progress"
            ))

        var_count = len(combinations[name])
        for idx, (i, r, sw) in enumerate(results):
            var_idx = idx // n_envs
            rewards[s][i].append(r)
            switchs[s][i].append(sw)

    with open("params/simulated_rewards_over_strategies.json", "w") as f:
        json.dump(rewards, f, indent=4)
    with open("params/simulated_switchs_over_strategies.json", "w") as f:
        json.dump(switchs, f, indent=4)
