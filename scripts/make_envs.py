import argparse
import os
import pickle

import torch
import matplotlib.pyplot as plt

from bandit25.envs import stationary, restless

parser = argparse.ArgumentParser(description="Generate data/envs/* and the sample env figures")
parser.add_argument("--process", choices=["daw", "original"], default="daw",
                    help="restless drift process, see bandit25.envs.restless")
args = parser.parse_args()
suffix = "" if args.process == "daw" else f"_{args.process}"

# same seed and order as the original envs.py; --process original rebuilds the old environments
torch.manual_seed(42)
stationary_noncor_envs = stationary()
restless_envs = restless(args.process)

os.makedirs("data/envs", exist_ok=True)
with open("data/envs/stationary_noncor_envs", 'wb') as f:
    pickle.dump(stationary_noncor_envs, f)
with open(f"data/envs/restless_envs{suffix}", 'wb') as f:
    pickle.dump(restless_envs, f)

plt.figure(figsize = (12, 5))
x = [i+1 for i in range(300)]
for t in range(4) :
    plt.plot(x, restless_envs[1][0].env[:, t].tolist(), linestyle='-')
plt.xlabel("horizon")
plt.ylabel("bandit mean")
plt.title(f"{restless_envs[1][0].name}")
plt.savefig(f"results/figures/sample env_{restless_envs[1][0].name}{suffix}.png")


ex = stationary_noncor_envs[1][0].env
plt.figure(figsize=(5, 2))
plt.title(f"{stationary_noncor_envs[1][0].name}")
ex = torch.reshape(ex, (1, 10))
plt.imshow(ex.tolist(), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[0, i].item():.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig(f'results/figures/sample env_{stationary_noncor_envs[1][0].name}.png')
