import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from function import simulate
from models import Kalman_SM, free_varaibles
import numpy as np

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)
    
name = 'Kalman_SM on restless envs'
envs = restless_envs[1]

param = []
for var_name in free_varaibles['Kalman_SM'] :
    param.append((torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item())

param = torch.tensor(param, dtype=torch.float64)
print(free_varaibles['Kalman_SM'])
print("[", param, "]")

print("<Simulation>")
al, rl, _, _ = simulate(param, envs[0], Kalman_SM, debug = 1, debug2=1)

print("<TEST>")
_, _, _, _, prob = simulate(param, envs[0], Kalman_SM, test = 1, real_actions=al, real_rewards=rl, debug = 1, debug2=1)
print("[Prob] ", prob)