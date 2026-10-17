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


def plot_fit(table, names, ranges, path, chance=0.25):
    """
        Fitted parameters of every subject, one panel each, and how well the model predicts the choices

        Args:
            table (DataFrame) : one row per subject with the columns in names, p_choice and switch_rate
            names (list of str) : parameter names
            ranges (dict) : name -> [lower, upper] bounds of the fit
            path (str) : where the figure is saved
            chance (float) : p_choice of random choices (1 / arms)
    """
    ncols = 6
    nrows = math.ceil((len(names) + 2) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = axes.flatten()
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, len(table))

    for ax, var in zip(axes, names):
        lo, hi = ranges[var]
        values = table[var].to_numpy()
        at_bound = (values - lo < 0.01 * (hi - lo)) | (hi - values < 0.01 * (hi - lo))
        ax.boxplot(values, widths=0.5, showfliers=False)
        ax.scatter(1 + jitter, values, s=12, c=np.where(at_bound, 'tab:red', 'black'), zorder=3)
        for b in (lo, hi):
            ax.axhline(b, color='gray', linestyle='--', linewidth=0.8)
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([])
        ax.set_title(var if not at_bound.any() else f"{var} ({at_bound.sum()} at bound)")

    # how well the fitted model predicts each subject's choices
    ax = axes[len(names)]
    p = np.sort(table["p_choice"].to_numpy())
    ax.bar(np.arange(len(p)), p, color='tab:blue')
    ax.axhline(chance, color='gray', linestyle='--', linewidth=0.8, label="chance")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("subjects (sorted)")
    ax.set_title("p(choice) per trial")

    ax = axes[len(names) + 1]
    ax.hist(table["switch_rate"], bins=10, range=(0, 1), color='tab:gray')
    ax.set_xlabel("switch rate")
    ax.set_title("observed switch rate")

    for ax in axes[len(names) + 2:]:
        ax.axis('off')

    fig.suptitle(f"{len(table)} subjects; red points sit at a bound of the fit")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_behavior(arm_means, actions, rewards, title, path):
    """
        One agent's choices on top of the drifting arm means

        Args:
            arm_means (array, (horizon, arms)) : true mean of every arm at every trial
            actions (array, (horizon,)) : chosen arm at every trial
            rewards (array, (horizon,)) : reward received at every trial
            title (str) : figure title, e.g. the agent's parameters
            path (str) : where the figure is saved
    """
    arm_means, actions, rewards = np.asarray(arm_means), np.asarray(actions), np.asarray(rewards)
    horizon, arms = arm_means.shape
    trials = np.arange(1, horizon + 1)
    colors = plt.get_cmap('tab10').colors[:arms]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    for a in range(arms):
        top.plot(trials, arm_means[:, a], color=colors[a], linewidth=1.2, label=f"arm {a}")
        chosen = actions == a
        top.scatter(trials[chosen], rewards[chosen], color=colors[a], s=8, alpha=0.7)
        bottom.scatter(trials[chosen], np.full(chosen.sum(), a), color=colors[a], marker='|', s=60)
    top.set_ylabel("arm mean (lines), reward (dots)")
    top.legend(loc='upper right', ncol=arms, fontsize=8)
    top.set_title(title, fontsize=10)
    bottom.set_yticks(range(arms))
    bottom.set_ylabel("chosen arm")
    bottom.set_xlabel("trial")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
