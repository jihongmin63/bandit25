import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from bandit25.variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from bandit25.function import simulate
from bandit25.models import Kalman_SM, free_varaibles
import numpy as np
import copy
from torchviz import make_dot
import time
start_time = time.time()

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("data/envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)
    
envs = restless_envs[1]
num = 30

bound = [[[recovery_range[var_name][0], recovery_range[var_name][1]]] for var_name in free_varaibles['Kalman_SM']]

original_params = []
recovered_params = []
for subj in tqdm(range(num)) :
    param = []
    for i, var_name in enumerate(free_varaibles['Kalman_SM']) :
        param.append((torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item())
    param = torch.tensor(param, dtype=torch.float64)
    original_params.append(param.tolist())
    recovered_params.append([])
    al, rl, _, _ = simulate(param, envs[subj], Kalman_SM)

    for selected_idx, selected_var in enumerate(free_varaibles['Kalman_SM']) :
        masked_param = copy.deepcopy(param)
        x0 = recovery_init[selected_var]

        def objective(x) :
            masked_param[selected_idx] = torch.tensor(x)
            _, _, _, _, prob = simulate(masked_param, envs[subj], Kalman_SM, test = 1, real_actions=al, real_rewards=rl)
            return -prob

        res = minimize(objective, x0, bounds = bound[selected_idx], method = 'L-BFGS-B')
        recovered_params[subj].append(res.x[0].tolist())

with open(f"results/params/Maksed_Restless_params", "wb") as f:
    pickle.dump([original_params, recovered_params], f)

end_time = time.time()