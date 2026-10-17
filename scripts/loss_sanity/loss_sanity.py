# 격자로 만들어서 멀어질수록 loss가 커지는지 확인
import argparse
import pickle

import torch
from tqdm import tqdm

from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_variables
from bandit25.fitting import bounds, sample_params

names = free_variables['Kalman_SM']
parser = argparse.ArgumentParser(description="Mean log-likelihood over a grid of (generating, evaluated) values of one parameter")
parser.add_argument("param", choices=names)
parser.add_argument("--grid", type=int, default=50)
args = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("data/envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)

envs = restless_envs[1]

num = 30
selected_envs = [envs[t] for t in range(num)]

idx = names.index(args.param)
lo, hi = bounds(names)
values = torch.linspace(lo[idx].item(), hi[idx].item(), args.grid, dtype=torch.float64)

# the other parameters are random per subject but shared by src and tgt
base = sample_params(names, num)

# every tgt value is evaluated on the same simulated data, in one batch of grid * num agents
tgt = base.repeat(1, args.grid, 1)
tgt[idx] = values.repeat_interleave(num).view(-1, 1)
tgt_envs = selected_envs * args.grid

# Z[i, j] : data generated with values[j] (src), evaluated at values[i] (tgt)
Z = torch.zeros(args.grid, args.grid, dtype=torch.float64)
for j in tqdm(range(args.grid), desc=args.param):
    src = base.clone()
    src[idx] = values[j]
    al, rl, _, _ = vectorized_simulate(src, selected_envs, Vectorized_Kalman_SM)
    with torch.no_grad():
        *_, prob = vectorized_simulate(tgt, tgt_envs, Vectorized_Kalman_SM, test=1,
                                       real_actions=[a.repeat(args.grid, 1) for a in al],
                                       real_rewards=[r.repeat(args.grid, 1) for r in rl])
    Z[:, j] = prob.view(args.grid, num).mean(dim=1)

# same layout as before: res[i * grid + j], read by loss_sanity_visualize.py
with open(f"results/params/Loss_sanity_{args.param}", "wb") as f:
    pickle.dump(list(Z.flatten()), f)
