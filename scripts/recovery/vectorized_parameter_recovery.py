import torch

import pickle

from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_varaibles
from bandit25.fitting import fit, sample_params, initial_params

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

x, log_lik = fit(initial_params(names, num), names, selected_envs, al, rl, desc="Fitting")

for i in range(num):
    print(f"{i}th params : " + " ".join(f"{x[j][i][0].item():.4f}" for j in range(len(names))) + f" | log-lik {log_lik[i]:.2f}")

with open("results/params/Vectorized_Restless_params", "wb") as f:
    pickle.dump([params, x], f)
