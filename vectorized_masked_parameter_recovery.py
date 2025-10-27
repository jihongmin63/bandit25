import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from vectorized_function import vectorized_simulate
from vectorized_models import Vectorized_Kalman_SM, free_varaibles
import numpy as np
import copy
from function import simulate
from models import Kalman_SM, free_varaibles

from torchviz import make_dot

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)
    
name = 'Kalman_SM on restless envs'
envs = restless_envs[1]

num = 30

selected_var = "alpha"

params = []
for n in range(num) :
    param = []
    for i, var_name in enumerate(free_varaibles['Kalman_SM']) :
        if var_name == selected_var : selected_idx = i
        param.append((torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item())
    params.append(param)
params = torch.tensor(params, dtype=torch.float64)
params = params.transpose(0, 1).unsqueeze(-1)

original_params = copy.deepcopy(params)

al, rl, _, _ = vectorized_simulate(params, [envs[t] for t in range(num)], Vectorized_Kalman_SM)

bound = [[recovery_range[selected_var][0], recovery_range[selected_var][1]] for i in range(num)]
x0 = []
for n in range(num) :
    x0.append([recovery_init[selected_var]])

params = params.numpy()
x0 = np.array(x0)
print(params[selected_idx].shape, x0.shape)

def objective(x) :
    params[selected_idx] = x.reshape(-1, 1)
    _, _, _, _, prob = vectorized_simulate(
        torch.from_numpy(params), [envs[t] for t in range(num)], Vectorized_Kalman_SM, test=1, real_actions=al, real_rewards=rl
    )
    return -prob

res = minimize(objective, x0, bounds = bound, method='L-BFGS-B')

with open(f"params/Vectorized_{selected_var}_maksed_Restless_params", "wb") as f:
    pickle.dump([original_params[selected_idx], res.x], f)

print(res.x)

"""
## nan error ##
nan_param = []
nan_param_recovered = x0_
print(params.size())
for i in range(len(free_varaibles['Kalman_SM'])) :
    nan_param.append(params[i][6][0])

print("7TH PARAMETER")
print(nan_param, nan_param_recovered)

print("TRAINING")

x = torch.nn.Parameter(
    torch.tensor(nan_param_recovered, requires_grad=True, dtype=torch.float64)
)
optimizer = torch.optim.Adam([x], lr=lr)
i = 6
al, rl, _, _ = simulate(nan_param, envs[i], Kalman_SM)
print(f"original parameter {nan_param}")
for time in tqdm(range(50), desc=f"Steps", leave=False):
    optimizer.zero_grad()
    _, _, _, _, prob = simulate(
        x, envs[i], Kalman_SM, test=1, real_actions=al, real_rewards=rl
    )
    loss = -prob.sum()
    loss.backward(retain_graph=True)

    optimizer.step()

    with torch.no_grad():
        for j, p in enumerate(x):
            p.clamp_(min=bound[j][0], max=bound[j][1])
    print(f"step {time} x : {x}")
    print(f"loss : {loss}")
print(x)
"""