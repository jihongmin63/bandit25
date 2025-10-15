import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from function import simulate
from models import Kalman_SM, free_varaibles
import numpy as np

from torchviz import make_dot

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)
    
name = 'Kalman_SM on restless envs'
envs = restless_envs[1]

param, bound = [], []
for var_name in free_varaibles['Kalman_SM'] :
    bound.append([recovery_range[var_name][0], recovery_range[var_name][1]])
    param.append((torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item())

param = torch.tensor(param, dtype=torch.float64,requires_grad=True)
print(free_varaibles['Kalman_SM'])
print("[", param, "]")

print("<Simulation>")
al, rl, _, _ = simulate(param, envs[0], Kalman_SM, debug = 1, debug2=1)

x0 = []
lr = 0.01
for var_name in free_varaibles['Kalman_SM']:
    x0.append(recovery_init[var_name])

recovered_x0 = []

x = torch.nn.Parameter(
    torch.tensor(x0, requires_grad=True, dtype=torch.float64)
)
optimizer = torch.optim.Adam([x], lr=lr)
for time in tqdm(range(100), desc=f"Steps", leave=False):
    optimizer.zero_grad()
    _, _, _, _, prob = simulate(
        x, envs[0], Kalman_SM, test=1, real_actions=al, real_rewards=rl
    )
    loss = -prob.sum()
    loss.backward(retain_graph=True)
    optimizer.step()

    with torch.no_grad():
        for j, p in enumerate(x):
            p.clamp_(min=bound[j][0], max=bound[j][1])
    print(f"step {time} x : {x}")
    print(f"loss : {loss}")

    recovered_x0.append(x.detach())

print("<TEST>")
_, _, _, _, prob = simulate(x, envs[0], Kalman_SM, test = 1, real_actions=al, real_rewards=rl, debug = 1, debug2=1)

#param_dict = {"loss" : param[0], "risk_plus" : param[1], "risk_minus" : param[2], "init_mean" : param[3], "init_noise" : param[4], "theta" : param[5], "alpha" : param[6], "rho" : param[7], "phi" : param[8], "xi" : param[9]}

#make_dot(prob, params=param_dict).render("torchviz", format="png")

print("[Prob] ", prob)
print(param, x)