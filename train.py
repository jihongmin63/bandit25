import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.optimize import minimize
import itertools
import optuna
import json
optuna.logging.set_verbosity(optuna.logging.WARNING)

from variable import variance, noise, horizon, num, mean, expected_mean, tot_env, env_name, strategy, variables, init, variable_init
from function import prospect, select_arm, get_action, simulate

# Env setting
np.random.seed(42)

# Agent simulation - Kalman Filter based
def reward(variables, option, e) :
    tot = 0

    for i in range(num) :
        cur = 0
        env = tot_env[e][i, :]

        _, _, cur, _ = simulate(variables, option, env)
            
        tot += cur
    
    return tot

params = {}
for st, name in enumerate(strategy) :
    def objective(trial):
        vari = []
        dic = variable_init[name]
        for n in variables :
            if n in dic :
                vari.append(trial.suggest_float(n, dic[n][0], dic[n][1]))
            else :
                vari.append(init[n])

        ret = 0
        for e in range(3) :
            vari[3] = mean[e]
            ret += reward(vari, st, e) - mean[e] * horizon * num
        
        return ret

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3000)

    print("[" + name + "]")
    print("Best parameters:", study.best_params)
    params[name] = study.best_params

with open("best_params.json", "w") as f:
    json.dump(params, f, indent=4)