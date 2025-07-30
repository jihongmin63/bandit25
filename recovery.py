import numpy as np
import json
import optuna
from multiprocessing import Pool, set_start_method, cpu_count
from optuna.samplers import TPESampler

from variable import horizon, tot_env, strategy, st_varaible, recovery_range, n_recovery, variables
from function import get_action, simulate

import warnings
from optuna._experimental import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def cross_entropy_loss(params, actions, rewards, envs, option):
    res = 0
    alpha, beta2, lam = 1, 1, 1

    if len(params) == 4:
        beta, mean, noise, temp = params
    elif len(params) == 5:
        beta, beta2, mean, noise, temp = params
    elif len(params) == 6:
        alpha, beta, lam, mean, noise, temp = params
    else:
        alpha, beta, beta2, lam, mean, noise, temp = params

    for num_e, env in enumerate(envs):
        result = 0
        mu_post = np.array([mean] * 10)
        sigma_post = np.array([noise] * 10)
        sigma_0 = noise
        epsilon = 1e-9

        for t in range(horizon):
            action, reward = actions[num_e][t], rewards[num_e][t]
            p = get_action(mu_post, sigma_post, option, result, alpha=alpha, beta=beta, beta2=beta2, lam=lam)
            p = np.exp((p - np.max(p)) / temp)
            p = p / np.sum(p)
            res -= np.log(p[action] + epsilon)

            result += reward
            K_t = sigma_post[action] / (sigma_post[action] + sigma_0)
            sigma_post[action] = (1 - K_t) * sigma_post[action]
            mu_post[action] += K_t * (reward - mu_post[action])

    return res


def build_objective(actions, rewards, envs, s, var_len, initial_params=None):
    def objective(trial):
        params = []
        for key in variables[var_len] :
            params.append(trial.suggest_float(key, recovery_range[key][0], recovery_range[key][1]))
        return cross_entropy_loss(params, actions, rewards, envs, s)

    return objective


def process_single_run(args):
    initial_param, s, name, envs = args
    actions, rewards, switches, recovered_switches = [], [], [], []
    for env in envs:
        a, r, _, switch = simulate(initial_param, s, env)
        actions.append(a)
        rewards.append(r)
        switches.append(switch)

    var_len = len(initial_param)
    objective = build_objective(actions, rewards, envs, s, var_len)
    
    sampler = TPESampler(multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=400, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    ordered = [best_params[k] for k in st_varaible[name]]

    for env in envs:
        _, _, _, recovered_switch = simulate(ordered, s, env)
        recovered_switches.append(recovered_switch)

    return ordered, switches, recovered_switches


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    envs = tot_env[1]
    
    for s, name in enumerate(strategy): 
        print(f"[{name} is currently working", end = " ")

        initial_params_list = []
        for _ in range(n_recovery):
            param = []
            for var_name in st_varaible[name]:
                param.append(np.random.uniform(recovery_range[var_name][0], recovery_range[var_name][1], 1)[0])
            initial_params_list.append(param)
        
        args_list = [(param, s, name, envs) for param in initial_params_list]

        with Pool(processes=cpu_count()) as pool:
            result_iter = pool.imap(process_single_run, args_list)
            recovered_results, initial_switches, recovered_switches = [], [], []

            for res, sw, re_sw in result_iter:
                recovered_results.append(res)
                initial_switches.append(sw)
                recovered_switches.append(re_sw)

        with open(f"params/parameter_recovery/recovered_params_{name}_optuna.json", "w") as f:
            json.dump([initial_params_list, recovered_results], f, indent=4)
        with open(f"params/parameter_recovery/recovered_switchs_{name}_optuna.json", "w") as f:
            json.dump([initial_switches, recovered_switches], f, indent=4)
        
        print("- finished]")
        print(np.array(initial_switches).shape, np.array(recovered_switches).shape)