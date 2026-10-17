import torch

import pickle

from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_variables
from bandit25.fitting import fit, sample_params

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("data/envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)

name = 'Kalman_SM on restless envs'
envs = restless_envs[1]
names = free_variables['Kalman_SM']

num = 30
initial_points = 10
selected_envs = [envs[t] for t in range(num)]

params = sample_params(names, num)
al, rl, _, _ = vectorized_simulate(params, selected_envs, Vectorized_Kalman_SM)

best_params, best_log_lik = None, None
for initial_idx in range(initial_points):
    # every subject starts from its own random point
    x, log_lik = fit(sample_params(names, num), names, selected_envs, al, rl,
                     desc=f"Initial point {initial_idx + 1}/{initial_points}")
    if best_params is None:
        best_params, best_log_lik = x, log_lik
    else:
        better = log_lik > best_log_lik
        best_params[:, better] = x[:, better]
        best_log_lik[better] = log_lik[better]
    print(f"{initial_idx}th initial point : mean best log-lik {best_log_lik.mean():.2f}")

for i in range(num):
    print(f"param {i} : " + " ".join(f"{best_params[j][i][0].item():.4f}" for j in range(len(names))) + f" | log-lik {best_log_lik[i]:.2f}")

# recovered params are stored as (num, n_params)
with open("results/params/Vectorized_Restless_params_w_initials", "wb") as f:
    pickle.dump([params, best_params.squeeze(-1).T], f)
