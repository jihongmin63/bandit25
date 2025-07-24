import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.optimize import minimize
import itertools
import json
from tqdm import tqdm

from variable import variance, noise, horizon, num, mean, expected_mean, tot_env, env_name, strategy, variables, recovery_set, combinations
from function import prospect, select_arm, get_action, simulate

rewards = [[[] for j in range(3)] for i in range(len(strategy))]
switchs = [[[] for j in range(3)] for i in range(len(strategy))]

for s, name in enumerate(strategy) :
    print("["+name+"]")
    for var in tqdm(combinations[name]) :
        for i in range(3) :
            r = []
            sw = []
            for j in range(num) :
                _, _, r_element, sw_element = simulate(var, s, tot_env[i][j, :])
                r.append(r_element)
                sw.append(sw_element)
            rewards[s][i].append(r)
            switchs[s][i].append(sw)

with open("params/simulated_rewards_over_strategies.json", "w") as f:
        json.dump(rewards, f, indent=4)
with open("params/simulated_switchs_over_strategies.json", "w") as f:
        json.dump(switchs, f, indent=4)