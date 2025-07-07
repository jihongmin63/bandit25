import numpy as np
import matplotlib.pyplot as plt

# Env setting

mean = 0
variance = 10
noise = 5
horizon = 20

np.random.seed(42)
env = np.random.normal(mean,variance,10)
def select_arm(n) : # select bandit w/ noise
    return env[n] + np.random.normal(0, noise)

plt.figure(figsize=(5, 2))
plt.title("sample multi armed bandit")
plt.imshow(env.reshape(1, -1), origin='lower', cmap = 'coolwarm')
for i in range(10) :
    plt.text(i, 0, f"{env[i]:.2f}", ha='center', va='center', color='black', fontsize=7)
plt.xticks([i for i in range(10)])
plt.yticks([])
plt.savefig('sample env.png')


# Agent simulation - Kalman Filter based
strategy = ['generic UCB', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)']
def simulate(option, hori = horizon) :
    al, rl = [], []
    result = 0

    mu_post = np.array([0 for i in range(10)])
    sigma_post = np.array([10 for i in range(10)])
    sigma_0 = 10

    for t in range(hori):
        action = get_action(mu_post, sigma_post, option, result)
        reward = select_arm(action)

        al.append(action)
        rl.append(reward)
        result += reward

        K_t = sigma_post[action] / (sigma_post[action] + sigma_0)
        sigma_post[action] = (1-K_t) * sigma_post[action]
        mu_post[action] += K_t * (reward - mu_post[action])

    return al, rl, result

def prospect(x, alpha=0.88, beta=0.88, lam=2.25):
    if isinstance(x, (float, int)):
        if x >= 0:
            return  np.power(x, alpha)
        else:
            return  -lam * np.power(-x, beta)
    elif isinstance(x, np.ndarray):
        util = np.empty_like(x, dtype=float)

        pos = x >= 0
        neg = x < 0

        util[pos] = np.power(x[pos], alpha)
        util[neg] = -lam * np.power(-x[neg], beta)

        return util

def get_action(mean, variance, option, result) :
    if (option == 0) : # generic UCB
        return np.argmax(mean + variance)
    elif (option == 1) : # relatvie prospect UCB
        return np.argmax(prospect(mean + result) - prospect(result) + variance)
    elif (option == 2) : # prospect UCB
        return np.argmax(prospect(mean) + variance)
    elif (option == 3) : # relative prospect UCB (inner)
        return np.argmax(prospect(mean + variance + result) - prospect(result))
 
    return 0


a, r, t = [], [], []
x = [i for i in range(horizon)]
for i in range(4) :
    print("[", strategy[i], "]")
    al, rl, result = simulate(i)
    print("actions", al)
    print("results", rl)
    print("total", result)
    print("")

    a.append(al)
    r.append(rl)
    t.append(result)

plt.figure(figsize=(6, 5))
plt.title('Reward accumulation over strategies')
for i in range(4) :
    plt.plot(x, np.cumsum(r[i]), linestyle='--', marker='o', label=strategy[i])
plt.xlabel("Trial")
plt.xticks([i for i in range(horizon)])
plt.ylabel("Accumulated Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('reward accumulation.png')

switch = [0 for i in range(4)]
for s in range(4) :
    for i in range(1, horizon) :
        if (a[s][i-1] != a[s][i]) :
            switch[s] += 1
    switch[s] /= (horizon-1)

plt.figure(figsize=(10, 5))
plt.title('# of Switches over strategies')
plt.bar([i for i in range(4)], switch, width = 0.6)
plt.xticks([i for i in range(4)], strategy)
for i, val in enumerate(switch):
    plt.text(i, val - 0.05, f"{val:.3f}", ha='center', va='bottom', fontsize=10, color="white")
plt.savefig('switch rate.png')