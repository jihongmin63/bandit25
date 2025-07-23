import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.optimize import minimize
import itertools
import json
from tqdm import tqdm

from variable import variance, noise, horizon, num, mean, expected_mean, tot_env, env_name, strategy, variables, recovery_set, combination
from function import prospect, select_arm, get_action, simulate

rewards = [[[] for j in range(3)] for i in range(len(strategy))]
switchs = [[[] for j in range(3)] for i in range(len(strategy))]
for s, name in enumerate(strategy) :
    print("["+name+"]")
    for var in tqdm(combination) :
        for i in range(3) :
            for j in range(num) :
                #print(var, i, j)
                _, _, r, sw = simulate(var, s, tot_env[i][j, :])
                rewards[s][i].append(r)
                switchs[s][i].append(sw)

with open("simulated_rewards_"+name+".json", "w") as f:
        json.dump(rewards, f, indent=4)
with open("simulated_switchs_"+name+".json", "w") as f:
        json.dump(switchs, f, indent=4)