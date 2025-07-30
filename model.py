import numpy as np
import json
import optuna
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from optuna.samplers import TPESampler

from variable import horizon, tot_env, strategy, st_varaible, recovery_range, n_recovery, variables
from function import get_action, simulate

import warnings
from optuna._experimental import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

def BIC_calculate(num_params, log_likelihoods, num_samples) :
    """
        Calculate BIC

        Args:
            num_params (int or int numpy array) 
            log_liklihoods (float or float numpy array)
            num_samples (int or int numpy array)

        Returns:
            BICs (float or float numpy array)
    """
    return -2 * log_likelihoods + num_params * np.log(num_samples)

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


def process_single_run(initial_param, s, name, envs):
    actions, rewards = [], []
    for env in envs:
        a, r, _, _ = simulate(initial_param, s, env)
        actions.append(a)
        rewards.append(r)

    num_params, num_samples, likelihoods = np.array([]), np.array([]), np.array([])
    
    # tqdm을 사용하여 외부 루프에 진행률 표시
    for current_s, current_name in enumerate(strategy): 
        var_len = len(st_varaible[current_name])
        objective = build_objective(actions, rewards, envs, current_s, var_len)

        sampler = TPESampler(multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=300, n_jobs=1, show_progress_bar=False) # n_jobs=1로 설정하여 병렬 처리 비활성화

        likelihoods = np.append(likelihoods, -study.best_value) # cross entropy loss가 아니라 log-likelihood 저장
        num_params = np.append(num_params, var_len)
        num_samples = np.append(num_samples, len(envs))

    return BIC_calculate(num_params, likelihoods, num_samples)



if __name__ == "__main__":
    envs = tot_env[1]

    for s, name in enumerate(strategy): 
        print(f"[{name} is currently working", end=" ")

        initial_params_list = []
        for _ in range(n_recovery):
            param = []
            for var_name in st_varaible[name]:
                param.append(np.random.uniform(recovery_range[var_name][0], recovery_range[var_name][1], 1)[0])
            initial_params_list.append(param)

        args_list = [(param, s, name, envs) for param in initial_params_list]

        with Pool(processes=8) as pool:
            recovered_results = pool.starmap(process_single_run, args_list)

        with open(f"params/model_recovery/recovered_models_{name}_optuna.json", "w") as f:
            json.dump([initial_params_list, recovered_results], f, indent=4)
        
        print("- finished]")