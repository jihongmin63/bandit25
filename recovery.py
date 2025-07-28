import numpy as np
import json
import optuna
from tqdm import tqdm
from multiprocessing import Pool, set_start_method, cpu_count

from variable import horizon, tot_env, strategy, st_varaible, recovery_range, n_recovery
from function import get_action, simulate

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
        suggest_args = {}
        if var_len == 4:
            suggest_args["beta"] = trial.suggest_float("beta", recovery_range["beta"][0], recovery_range["beta"][1])
            suggest_args["mean"] = trial.suggest_float("mean", recovery_range["mean"][0], recovery_range["mean"][1])
            suggest_args["noise"] = trial.suggest_float("noise", recovery_range["noise"][0], recovery_range["noise"][1])
            suggest_args["temp"] = trial.suggest_float("temp", recovery_range["temp"][0], recovery_range["temp"][1])
            params = [suggest_args["beta"], suggest_args["mean"], suggest_args["noise"], suggest_args["temp"]]
        elif var_len == 5:
            suggest_args["beta"] = trial.suggest_float("beta", recovery_range["beta"][0], recovery_range["beta"][1])
            suggest_args["beta2"] = trial.suggest_float("beta2", recovery_range["beta2"][0], recovery_range["beta2"][1])
            suggest_args["mean"] = trial.suggest_float("mean", recovery_range["mean"][0], recovery_range["mean"][1])
            suggest_args["noise"] = trial.suggest_float("noise", recovery_range["noise"][0], recovery_range["noise"][1])
            suggest_args["temp"] = trial.suggest_float("temp", recovery_range["temp"][0], recovery_range["temp"][1])
            params = [suggest_args["beta"], suggest_args["beta2"], suggest_args["mean"], suggest_args["noise"], suggest_args["temp"]]
        elif var_len == 6:
            suggest_args["alpha"] = trial.suggest_float("alpha", recovery_range["alpha"][0], recovery_range["alpha"][1])
            suggest_args["beta"] = trial.suggest_float("beta", recovery_range["beta"][0], recovery_range["beta"][1])
            suggest_args["lambda"] = trial.suggest_float("lambda", recovery_range["lambda"][0], recovery_range["lambda"][1])
            suggest_args["mean"] = trial.suggest_float("mean", recovery_range["mean"][0], recovery_range["mean"][1])
            suggest_args["noise"] = trial.suggest_float("noise", recovery_range["noise"][0], recovery_range["noise"][1])
            suggest_args["temp"] = trial.suggest_float("temp", recovery_range["temp"][0], recovery_range["temp"][1])
            params = [suggest_args["alpha"], suggest_args["beta"], suggest_args["lambda"], suggest_args["mean"], suggest_args["noise"], suggest_args["temp"]]
        else: # len(params) == 7
            suggest_args["alpha"] = trial.suggest_float("alpha", recovery_range["alpha"][0], recovery_range["alpha"][1])
            suggest_args["beta"] = trial.suggest_float("beta", recovery_range["beta"][0], recovery_range["beta"][1])
            suggest_args["beta2"] = trial.suggest_float("beta2", recovery_range["beta2"][0], recovery_range["beta2"][1])
            suggest_args["lambda"] = trial.suggest_float("lambda", recovery_range["lambda"][0], recovery_range["lambda"][1])
            suggest_args["mean"] = trial.suggest_float("mean", recovery_range["mean"][0], recovery_range["mean"][1])
            suggest_args["noise"] = trial.suggest_float("noise", recovery_range["noise"][0], recovery_range["noise"][1])
            suggest_args["temp"] = trial.suggest_float("temp", recovery_range["temp"][0], recovery_range["temp"][1])
            params = [suggest_args["alpha"], suggest_args["beta"], suggest_args["beta2"], suggest_args["lambda"], suggest_args["mean"], suggest_args["noise"], suggest_args["temp"]]

        return cross_entropy_loss(params, actions, rewards, envs, s)

    return objective


def process_single_run(args):
    initial_param, s, name, envs = args
    actions, rewards = [], []
    for env in envs:
        a, r, _, _ = simulate(initial_param, s, env)
        actions.append(a)
        rewards.append(r)

    var_len = len(initial_param)
    objective = build_objective(actions, rewards, envs, s, var_len)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    ordered = [best_params[k] for k in st_varaible[name]]

    return ordered


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    envs = tot_env[1]
    
    for s, name in enumerate(strategy): 
        print(f"[{name}]")

        initial_params_list = []
        for _ in range(n_recovery):
            param = []
            for var_name in st_varaible[name]:
                param.append(np.random.uniform(recovery_range[var_name][0], recovery_range[var_name][1], 1)[0])
            initial_params_list.append(param)
        
        args_list = [(param, s, name, envs) for param in initial_params_list]

        with Pool(processes=cpu_count()) as pool:
            recovered_results = list(tqdm(pool.imap_unordered(process_single_run, args_list), total=len(args_list)))

        with open(f"params/recovered_params_{name}_optuna.json", "w") as f:
            json.dump([initial_params_list, recovered_results], f, indent=4)