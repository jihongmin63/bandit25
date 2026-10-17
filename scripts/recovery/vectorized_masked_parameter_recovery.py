import torch

import pickle

from bandit25.variable import recovery_init
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_varaibles
from bandit25.fitting import fit, sample_params

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("data/envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)

name = 'Kalman_SM on restless envs'
envs = restless_envs[1]
names = free_varaibles['Kalman_SM']

num = 30
selected_envs = [envs[t] for t in range(num)]

params = sample_params(names, num)
al, rl, _, _ = vectorized_simulate(params, selected_envs, Vectorized_Kalman_SM)

for selected_idx, selected_var in enumerate(names):
    # only selected_var is fitted; every other parameter stays at its true value
    x0 = params.clone()
    x0[selected_idx] = recovery_init[selected_var]
    x, _ = fit(x0, names, selected_envs, al, rl, free=[selected_var], desc=selected_var)
    print(f"{selected_var} : {x[selected_idx, :, 0].tolist()}")

    with open(f"results/params/Vectorized_{selected_var}_maksed_Restless_params", "wb") as f:
        pickle.dump([params[selected_idx, :, 0], x[selected_idx, :, 0]], f)
