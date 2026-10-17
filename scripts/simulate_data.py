import argparse
import os
import pickle

import pandas as pd
import torch

from bandit25.data import save_choices
from bandit25.fitting import sample_params
from bandit25.plotting import plot_behavior
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_variables

parser = argparse.ArgumentParser(description="Simulate agents and write their choices in the CSV format of fit_data.py")
parser.add_argument("--n", type=int, default=20, help="number of simulated subjects")
parser.add_argument("--envs", default="data/envs/restless_envs", help="made by scripts/make_envs.py")
parser.add_argument("--out", default="data/simulated", help="directory for choices.csv and true_params.csv")
parser.add_argument("--figure", default="results/figures/sample_behavior.png", help="choices of the first subject")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
names = free_variables['Kalman_SM']

with open(args.envs, 'rb') as f:
    envs = pickle.load(f)[1][:args.n]  # the 'mix' environments, centered at 0

params = sample_params(names, args.n)
al, rl, _, _ = vectorized_simulate(params, envs, Vectorized_Kalman_SM)

os.makedirs(args.out, exist_ok=True)
save_choices(os.path.join(args.out, "choices.csv"), al, rl)
true = pd.DataFrame(params[:, :, 0].T.numpy(), columns=names)
true.insert(0, "subject", range(args.n))
true.to_csv(os.path.join(args.out, "true_params.csv"), index=False)

p0 = ", ".join(f"{v}={params[i, 0, 0].item():.2f}" for i, v in enumerate(names))
plot_behavior(envs[0].env, torch.stack(al)[:, 0].argmax(dim=-1), torch.stack(rl)[:, 0, 0],
              f"simulated subject 0: {p0}", args.figure)

print(f"wrote {args.out}/choices.csv and {args.out}/true_params.csv ({args.n} subjects, "
      f"{len(al)} trials, task center {envs[0].mean}) and {args.figure}")
