import argparse
import math
import os

import pandas as pd
import torch

from bandit25.data import load_choices, task_envs
from bandit25.envs import lamb, noise_o, noise_d
from bandit25.fitting import fit, initial_params
from bandit25.plotting import plot_fit, plot_recovery
from bandit25.variable import recovery_range
from bandit25.vectorized_models import free_variables

parser = argparse.ArgumentParser(description="Fit Kalman_SM to observed choices (see README, 'Fitting real data')")
parser.add_argument("csv", help="columns subject, trial, choice (0-3), reward")
parser.add_argument("--center", type=float, required=True, help="decay center of the arm means in the task")
parser.add_argument("--noise", type=float, default=noise_o, help="observation noise of the task (sigma_o)")
parser.add_argument("--noise-d", type=float, default=noise_d, help="drift noise of the task (sigma_d)")
parser.add_argument("--lamb", type=float, default=lamb, help="decay of the arm means (lambda)")
parser.add_argument("--max-eval", type=int, default=200, help="likelihood evaluations of L-BFGS")
parser.add_argument("--out", help="output directory (default: results/fits/<csv name>)")
parser.add_argument("--true", help="CSV of true parameters (simulated data): also plots recovery")
args = parser.parse_args()

out = args.out or os.path.join("results/fits", os.path.splitext(os.path.basename(args.csv))[0])
os.makedirs(out, exist_ok=True)
names = free_variables['Kalman_SM']

subjects, actions, rewards = load_choices(args.csv)
n, horizon = len(subjects), len(actions)
envs = task_envs(n, horizon, args.center, noise=args.noise, noise_d=args.noise_d, lamb=args.lamb)
print(f"{n} subjects x {horizon} trials from {args.csv}")

x, log_lik = fit(initial_params(names, n), names, envs, actions, rewards, max_eval=args.max_eval, desc="Fitting")

table = pd.DataFrame(x[:, :, 0].T.numpy(), columns=names)
table.insert(0, "subject", subjects)
table["log_lik"] = log_lik.numpy()
table["p_choice"] = torch.exp(log_lik / horizon).numpy()        # geometric mean probability of the observed choices
table["bic"] = (len(names) * math.log(horizon) - 2 * log_lik).numpy()
choices = torch.stack(actions)
table["switch_rate"] = (choices[1:] != choices[:-1]).any(dim=-1).double().mean(dim=0).numpy()

at_bound = pd.DataFrame({v: (table[v] - recovery_range[v][0] < 0.01 * (recovery_range[v][1] - recovery_range[v][0]))
                            | (recovery_range[v][1] - table[v] < 0.01 * (recovery_range[v][1] - recovery_range[v][0]))
                         for v in names})
table["at_bound"] = at_bound.apply(lambda row: " ".join(v for v in names if row[v]), axis=1)
table.to_csv(os.path.join(out, "params.csv"), index=False)
plot_fit(table, names, recovery_range, os.path.join(out, "fit.png"), chance=1 / actions[0].size(-1))

print(f"mean log-lik {table['log_lik'].mean():.1f}, median p(choice) {table['p_choice'].median():.2f} "
      f"(chance {1 / actions[0].size(-1):.2f}), total BIC {table['bic'].sum():.1f}")
counts = at_bound.sum()
if counts.any():
    print("fits at a bound of recovery_range: " + ", ".join(f"{v} {c}" for v, c in counts.items() if c))

if args.true:
    true = pd.read_csv(args.true).set_index("subject").loc[subjects]
    plot_recovery(true[names].to_numpy().T, table[names].to_numpy().T, names, os.path.join(out, "recovery.png"))

print(f"wrote {out}/params.csv and {out}/fit.png" + (f" and {out}/recovery.png" if args.true else ""))
