import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from bandit25.variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_varaibles
import numpy as np


from bandit25.function import simulate
from bandit25.models import Kalman_SM, free_varaibles

from torchviz import make_dot

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("data/envs/restless_envs", 'rb') as f:
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
    print(prob.requires_grad)
    loss = -prob.sum()
    for j in tqdm(range(num), desc="Backwarding", position=1, leave=False):
        loss[j].backward(retain_graph=(j != num - 1))
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

with open("results/params/Vectorized_Restless_params", "wb") as f:
    pickle.dump([params, x], f)