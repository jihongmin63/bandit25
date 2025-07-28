import numpy as np
import json
import optuna
from tqdm import tqdm

from variable import variance, noise, horizon, num, mean, expected_mean, tot_env, env_name, strategy, variables, st_varaible, recovery_range
from function import prospect, select_arm, get_action, simulate

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
            p = get_action(mu_post, sigma_post, option, result, alpha=alpha, beta=beta, lam=lam)
            p = np.exp((p - np.max(p)) / temp)
            p = p / np.sum(p)
            res -= np.log(p[action] + epsilon)

            result += reward
            K_t = sigma_post[action] / (sigma_post[action] + sigma_0)
            sigma_post[action] = (1 - K_t) * sigma_post[action]
            mu_post[action] += K_t * (reward - mu_post[action])

    return res


def build_objective(actions, rewards, envs, s, var_len):
    def objective(trial):
        if var_len == 4:
            beta = trial.suggest_float("beta", 0, 10)
            mean = trial.suggest_float("mean", -100, 100)
            noise = trial.suggest_float("noise", 0.1, 20)
            temp = trial.suggest_float("temp", 0.01, 10)
            params = [beta, mean, noise, temp]
        elif var_len == 5:
            beta = trial.suggest_float("beta", 0, 10)
            beta2 = trial.suggest_float("beta2", 0, 10)
            mean = trial.suggest_float("mean", -100, 100)
            noise = trial.suggest_float("noise", 0.1, 20)
            temp = trial.suggest_float("temp", 0.01, 10)
            params = [beta, beta2, mean, noise, temp]
        elif var_len == 6:
            alpha = trial.suggest_float("alpha", 0.1, 2.0)
            beta = trial.suggest_float("beta", 0, 10)
            lam = trial.suggest_float("lambda", 0, 10)
            mean = trial.suggest_float("mean", -100, 100)
            noise = trial.suggest_float("noise", 0.1, 20)
            temp = trial.suggest_float("temp", 0.01, 10)
            params = [alpha, beta, lam, mean, noise, temp]
        else:
            alpha = trial.suggest_float("alpha", 0.1, 2.0)
            beta = trial.suggest_float("beta", 0, 10)
            beta2 = trial.suggest_float("beta2", 0, 10)
            lam = trial.suggest_float("lambda", 0, 10)
            mean = trial.suggest_float("mean", -100, 100)
            noise = trial.suggest_float("noise", 0.1, 20)
            temp = trial.suggest_float("temp", 0.01, 10)
            params = [alpha, beta, beta2, lam, mean, noise, temp]

        return cross_entropy_loss(params, actions, rewards, envs, s)

    return objective


def process_single_var(var, s, name, envs):
    actions, rewards = [], []
    for env in envs:
        a, r, _, _ = simulate(var, s, env)
        actions.append(a)
        rewards.append(r)

    var_len = len(var)
    objective = build_objective(actions, rewards, envs, s, var_len)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    ordered = [best_params[k] for k in st_varaible[name]]

    return ordered


if __name__ == "__main__":
    envs = tot_env[1]
    for s, name in enumerate(strategy):
        print(f"[{name}]")
        
        params, results = [], []
        for i in tqdm(range(100)) :
            param = []
            for var_name in st_varaible[name] :
                param.append(np.random.normal(recovery_range[var_name][0], recovery_range[var_name][1], 1)[0])
            params.append(param)

            result = process_single_var(param, s, name, envs)
            results.append(result)

        with open(f"params/recovered_params_{name}_optuna.json", "w") as f:
            json.dump([params, results], f, indent=4)