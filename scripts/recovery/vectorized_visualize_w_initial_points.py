import pickle

import torch

from bandit25.vectorized_models import free_variables
from bandit25.plotting import plot_recovery

with open("results/params/Vectorized_Restless_params_w_initials","rb") as f:
    original_data, recovered_data = pickle.load(f)

# original (n_params, num, 1), recovered (num, n_params)
plot_recovery(torch.as_tensor(original_data).detach()[..., 0], torch.as_tensor(recovered_data).detach().T,
              free_variables['Kalman_SM'], "results/figures/recovery/vectorized_all_parameters_w_initial_points.png")
