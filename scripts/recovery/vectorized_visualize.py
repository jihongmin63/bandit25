import pickle

import torch

from bandit25.vectorized_models import free_variables
from bandit25.plotting import plot_recovery

with open("results/params/Vectorized_Restless_params","rb") as f:
    original_data, recovered_data = pickle.load(f)

# both (n_params, num, 1)
plot_recovery(torch.as_tensor(original_data).detach()[..., 0], torch.as_tensor(recovered_data).detach()[..., 0],
              free_variables['Kalman_SM'], "results/figures/recovery/vectorized_all_parameters.png")
