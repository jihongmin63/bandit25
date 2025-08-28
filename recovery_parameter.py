import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize

import pickle

from variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from function import simulate
from models import Kalman_SM, free_varaibles


def cross_entropy_loss(params, real_actions_list, real_rewards_list, envs):
    res = 0

    for env, actions, rewards in enumerate(zip(envs, real_actions_list, real_rewards_list)):
        _, _, _, _, prob = simulate(params, env, test = 1, real_actions=actions, real_rewards=rewards)
        rest += prob
    
    return -res



if __name__ == "__main__":
    with open("envs/restless_envs", 'rb') as f:
        restless_envs = pickle.load(f)
    
    name = 'Kalman_SM on restless envs'
    envs = restless_envs[1]
    
    initial_params_list, recovered_params_list = [], []
    for _ in range(recovery_subjects):
        param = []
        x0 = []
        bounds = []
        for var_name in free_varaibles['Kalman_SM'] :
                param.append((torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item())
                bounds.append(recovery_range[var_name])
                x0.append(recovery_init[var_name])
        initial_params_list.append(param)
            
        real_actions, real_rewards = [], []
        for i in range(recovery_trials) :
            al, rl, _, _, _ = simulate(param, envs[i], Kalman_SM)
            real_actions.append(al)
            real_rewards.append(rl)
        
        recovered = minimize(cross_entropy_loss, x0=x0, bounds=bounds, args = (real_actions, real_rewards, envs[:recovery_trials]))
        recovered_params_list.append(recovered)

        with open(f"params/parameter_recovery/recovered_params_{name}.json", 'wb') as f:
            pickle.dump([initial_params_list, recovered_params_list], f)