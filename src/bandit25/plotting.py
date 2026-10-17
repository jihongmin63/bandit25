import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_recovery(original, recovered, names, path):
    """
        Original vs recovered value of every parameter, one panel each

        Args:
            original, recovered (array, (n_params, n_subjects)) : parameter values
            names (list of str) : parameter names, in the order of the rows
            path (str) : where the figure is saved
    """
    original, recovered = np.asarray(original), np.asarray(recovered)
    ncols = 5
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, var in enumerate(names):
        ax = axes[idx]
        orig_col, rec_col = f"original {var}", f"recovered {var}"
        df = pd.DataFrame({orig_col: original[idx], rec_col: recovered[idx]})

        try:
            sns.kdeplot(data=df, x=orig_col, y=rec_col, fill=True, levels=10, ax=ax, warn_singular=False)
        except Exception:
            pass
        # the kde spreads beyond the parameter bounds, so show the points as well
        sns.scatterplot(data=df, x=orig_col, y=rec_col, ax=ax, color='black', s=12)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')

        r = np.corrcoef(df[orig_col], df[rec_col])[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=10, verticalalignment='top')

        ax.set_title(f"{var}")

    for ax in axes[len(names):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
