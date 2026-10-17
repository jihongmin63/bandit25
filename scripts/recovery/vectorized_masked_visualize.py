import pickle

import torch

from bandit25.vectorized_models import free_variables
from bandit25.plotting import plot_recovery

names = free_variables['Kalman_SM']
original, recovered = [], []
for idx, var in enumerate(names):
    with open(f"results/params/Vectorized_{var}_masked_Restless_params","rb") as f:
        original_data, recovered_data = pickle.load(f)

    # runs before d738434 saved all parameters (n_params, num, 1) or a numpy array
    original_data = torch.as_tensor(original_data).detach()
    if original_data.dim() == 3:
        original_data = original_data[idx]
    original.append(original_data.reshape(-1))
    recovered.append(torch.as_tensor(recovered_data).detach().reshape(-1))

plot_recovery(torch.stack(original), torch.stack(recovered), names,
              "results/figures/recovery/vectorized_masked_all_parameters.png")
