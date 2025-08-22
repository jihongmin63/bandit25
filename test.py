import numpy as np
import json
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import minimize
import pickle
import matplotlib.pyplot as plt

from variable import Env
from models import Kalman_SM, Kalman_PMU, Decay_SM
from function import prospect, select_arm, simulate

with open("envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)


test_env = restless_envs[1][:50]

# loss 정의
def objective(x) :
    res = 0
    for env in test_env :
        _, _, r, _, _ = simulate(x, env, Kalman_PMU)
        res += r
    return -res

# self.loss, self.risk, self.mean, self.noise, self.theta1, self.theta2, self.kappa, self.alpha, self.rho, self.beta1, self.beta2, self.phi2, self.phi3
#sample = [1, 1, 0, 1000, 0.33, 0, 1, 0, 0, 0.54, 0.54, 0, 0]
#initial_guess = [1, 1, 50, 4, 10, 10, 0.5, 0.5, 1, 30, 30, 20, 20]  
#bounds = [(0, 5), (0, 2), (-50, 50), (0, float('inf')), (0, 100), (0, 100), (0, 1), (0, 1), (0, 30), (-5, 5), (-5, 5), (0, 100), (0, 100)] 
initial_guess = [0.22, 0.61, 0, 4]
bounds = [(0, 5), (0, 2), (-50, 50), (0, 100)]
result = minimize(objective, initial_guess, bounds=bounds)
print(result.x) 

# simulate
al, rl, _, _, _ = simulate(result.x, test_env[0], Kalman_PMU)
#print(al)

plt.plot(range(len(rl)), rl, linestyle='-')
plt.savefig('simulation.png')