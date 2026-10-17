import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from bandit25.variable import recovery_range
from bandit25.vectorized_models import free_variables

# every parameter that loss_sanity.py has been run for
names = [v for v in free_variables['Kalman_SM'] if os.path.exists(f"results/params/Loss_sanity_{v}")]
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.4), squeeze=False, constrained_layout=True)

for ax, var in zip(axes[0], names):
    with open(f"results/params/Loss_sanity_{var}", 'rb') as f:
        res = pickle.load(f)
    z = torch.stack([torch.as_tensor(r).flatten() for r in res]).flatten().numpy()
    grid = int(round(len(z) ** 0.5))
    Z = z.reshape(grid, grid)  # Z[i, j] : data generated with values[j], evaluated at values[i]

    lo, hi = recovery_range[var]
    values = np.linspace(lo, hi, grid)
    # each column is its own dataset, so compare values within a column only
    rel = Z - Z.max(axis=0, keepdims=True)
    best = values[Z.argmax(axis=0)]

    im = ax.imshow(rel, origin='lower', extent=[lo, hi, lo, hi], aspect='auto', cmap='viridis', vmin=-10, vmax=0)
    ax.plot([lo, hi], [lo, hi], color='white', linestyle='--', linewidth=1)
    ax.scatter(values, best, s=8, color='red', zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"generating {var}")
    ax.set_ylabel(f"evaluated {var}")
    ax.set_title(f"{var} (r = {np.corrcoef(values, best)[0, 1]:.2f})")

fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85,
             label="mean log-lik per subject, minus the best in the column")
fig.suptitle("Loss sanity: red = best evaluated value for each generating value (ideal: the dashed diagonal)")
plt.savefig("results/figures/loss_sanity.png", dpi=200)
plt.close()
