import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from vectorized_function import vectorized_simulate
from vectorized_models import Vectorized_Kalman_SM, free_varaibles
import numpy as np


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

params = []
for n in range(num) :
    param, bound = [], []
    for var_name in free_varaibles['Kalman_SM'] :
        bound.append([recovery_range[var_name][0], recovery_range[var_name][1]])
        param.append((torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item())
    params.append(param)
params = torch.tensor(params, dtype=torch.float64,requires_grad=True)

params = params.transpose(0, 1).unsqueeze(-1)
al, rl, _, _ = vectorized_simulate(params, [envs[t] for t in range(num)], Vectorized_Kalman_SM)

x0_ = []
lr = 0.01
for var_name in free_varaibles['Kalman_SM']:
    x0_.append(recovery_init[var_name])
x0 = [x0_ for i in range(num)]

recovered_x0 = []
x = torch.nn.Parameter(
    torch.tensor(x0, requires_grad=True, dtype=torch.float64).transpose(0, 1).unsqueeze(-1)
)
optimizer = torch.optim.Adam([x], lr=lr)
for time in tqdm(range(50), desc=f"Steps", leave=False):
    optimizer.zero_grad()
    _, _, _, _, prob = vectorized_simulate(
        x, [envs[t] for t in range(num)], Vectorized_Kalman_SM, test=1, real_actions=al, real_rewards=rl
    )
    loss = -prob.sum()
    loss.backward(retain_graph=True)
    optimizer.step()
    # bound
    with torch.no_grad():
        for j, p in enumerate(x):
            p.clamp_(min=bound[j][0], max=bound[j][1])

    print(f"step {time}")
    for i in range(num):
        print(f"{i}th params : ", end = "")
        for j, _ in enumerate(x) :
            print(f"{x[j][i][0].item()} ", end = "")
        print("")
    print("")

with open("params/Vectorized_Restless_params", "wb") as f:
    pickle.dump([params, x], f)

print("Vector Recovered")
## nan error ##
nan_param = []
nan_param_recovered = x0_
print(params.size())
for i in range(len(free_varaibles['Kalman_SM'])) :
    nan_param.append(params[i][6][0])
    print(x[i][6][0], end = " ")
print("")

x = torch.nn.Parameter(
    torch.tensor(nan_param_recovered, requires_grad=True, dtype=torch.float64)
)
optimizer = torch.optim.Adam([x], lr=lr)
i = 6
al, rl, _, _ = simulate(nan_param, envs[i], Kalman_SM)
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
    #print(f"step {time} x : {x}")
    #print(f"loss : {loss}")
print("recovered")
print(x)
