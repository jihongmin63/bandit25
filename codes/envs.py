import torch
import matplotlib.pyplot as plt
import pickle
from .variable import Env, mean, env_name

variance = 10
noise = 10
horizon = 70
num = 100

torch.manual_seed(42)

# stationary, non-correlated
def stationary() :
    stationary_noncor_envs = [[] for _ in range(3)]
    for i in range(3) :
        for _ in range(100) :
            e = Env(env = torch.normal(mean[i], noise, size=(10,)), stationary = 1, structure = (10,), horizon = horizon, mean = mean[i], noise = noise, name = f"stationary_noncor_{env_name[i]}")
            stationary_noncor_envs[i].append(e)
    return stationary_noncor_envs
"""
with open("envs/stationary_noncor_envs", 'wb') as f:
    pickle.dump(stationary_noncor_envs, f)

# restless
lamb, noise_o, noise_d = 0.9836, 4, 2.8
restless_envs = [[] for _ in range(3)]
for i in range(3) :
    for _ in range(100) :
        theta = float(mean[i])
        env = [[torch.normal(theta, 10.0, size=(1,))] for _ in range(4)]

        for t in range(4):
            for h in range(1, 300):
                mu_post = torch.normal(env[t][-1], noise_o)
                mu_pre = torch.normal(lamb * mu_post + (1 - lamb) * theta, noise_d)
                env[t].append(mu_pre)

        env = torch.transpose(torch.Tensor(env), 0, 1)

        e = Env(env = env, stationary = 0, structure = (4,), horizon = 300, mean = theta, noise = noise_o, noise_d = noise_d, lamb = lamb, name = f"restless_{env_name[i]}")
        restless_envs[i].append(e)

with open("envs/restless_envs", 'wb') as f:
    pickle.dump(restless_envs, f)

plt.figure(figsize = (12, 5))
x = [i+1 for i in range(300)]
for t in range(4) :
    plt.plot(x, restless_envs[1][0].env.T[t].tolist(), linestyle='-')
plt.xlabel("horizon")
plt.ylabel("bandit mean")
plt.title(f"{restless_envs[1][0].name}")
plt.savefig(f"images/sample env_{restless_envs[1][0].name}.png")


ex = stationary_noncor_envs[1][0].env
plt.figure(figsize=(5, 2))
plt.title(f"{stationary_noncor_envs[1][0].name}")
ex = torch.reshape(ex, (1, 10))
plt.imshow(ex.tolist(), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[0, i].item():.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig(f'images/sample env_{stationary_noncor_envs[1][0].name}.png')
"""
"""
# corelated 1D
d, l = 10, [1, 2]
name = ['hard', 'soft']
stationary_cor_1D_envs = [[[] for _ in name] for _ in mean]
for i, mu in enumerate(mean) :
    for j, cor in enumerate(l) :
        for _ in range(100) :
            mean_vector = np.ones(d)*mu
            covariance_vector = np.array([
                [np.exp(-0.5 * ((x_i-x_j)/cor)**2) for x_j in range(d)]
                for x_i in range(d)
            ])*50+0.01*np.identity(d)

            sample = np.random.multivariate_normal(mean_vector, covariance_vector)
            sample = np.sum(sample, np.random.normal(0, 1, size=sample.shape))
            stationary_cor_1D_envs[i][j].append(Env(env = sample, stationary = 1, structure = (d,), horizon = 25, mean = mu, noise = , l = cor, name = f"stationary_cor_1D_{name[j]}_{env_name[i]}"))

with open("envs/stationary_cor_1D_envs", 'wb') as f:
    pickle.dump(stationary_cor_1D_envs, f)

# corelated 2D
stationary_cor_2D_envs = [[[] for _ in name] for _ in mean]
for i, mu in enumerate(mean) :
    for j, cor in enumerate(l) :
        for _ in range(100) :
            mean_vector = np.ones(d**2)*mu
            covariance_vector = np.array([
                [np.exp(-0.5 * ((x_i//d-x_j//d)**2+(x_i%d-x_j%d)**2)/cor**2) for x_j in range(d**2)]
                for x_i in range(d**2)
            ])*50+0.01*np.identity(d**2)

            sample = np.random.multivariate_normal(mean_vector, covariance_vector).reshape(d, d)
            sample = np.sum(sample, np.random.normal(0, 1, size=sample.shape))
            stationary_cor_2D_envs[i][j].append(Env(env = sample, stationary = 1, structure = (d,d,), horizon = 25, mean = mu, noise = , l = cor, name = f"stationary_cor_2D_{name[j]}_{env_name[i]}"))

with open("envs/stationary_cor_2D_envs", 'wb') as f:
    pickle.dump(stationary_cor_2D_envs, f)


# env sample images

ex = stationary_cor_1D_envs[1][1][0].env
plt.figure(figsize=(5, 2))
plt.title(f"{stationary_cor_1D_envs[1][1][0].name}")
plt.imshow(ex.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{ex[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig(f'images/sample env_{stationary_cor_1D_envs[1][1][0].name}.png')


ex = stationary_cor_2D_envs[1][1][0].env
plt.figure(figsize=(5, 5))
plt.title(f"{stationary_cor_2D_envs[1][1][0].name}")
plt.imshow(ex, origin='lower', cmap='coolwarm')
for i in range(10):
    for j in range(10):
        plt.text(j, i, f"{ex[i][j]:.2f}", ha='center', va='center',
                 color='black', fontsize=7)
plt.xticks(range(10))
plt.yticks(range(10))
plt.savefig(f'images/sample env_{stationary_cor_2D_envs[1][1][0].name}.png')
"""