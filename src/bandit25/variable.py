import numpy as np
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass
import pickle

np.random.seed(42)

@dataclass
class Env:
    env: np.ndarray         # num x structure (x horizon if restless)
    stationary : int        # 1 -> stationary, 0 -> restless
    structure: tuple = None # structure of bandit in each env
    horizon: int = None     # number of horizon in each task
    mean: float = None        # mean of bandits & decay center (restless)
    noise: float = None       # noise of bandits (stationary) & obsevational noise of bandits (restless)
    noise_d : float = None    # step noise of bandits (restless)
    lamb : int = None       # decay parameter (restless)
    l : int = None          # correlated value (correlated)
    name: str = None

variance = 10
noise = 10
horizon = 70
num = 100
mean = [100, 0, -100]
env_name = ['pos', 'mix', 'neg']

recovery_range = {
    'loss' : [0.0, 10.0], 
    'risk_plus' : [0.0, 3.0], 
    'risk_minus' : [0.0, 3.0],  
    'init_mean' : [-100.0, 100.0], 
    'init_noise' : [2.0, 10.0], 
    'theta' : [1e-4, 5.0], 
    'alpha' : [0.0, 1.0], 
    'rho' : [-20.0, 20.0], 
    'phi' : [-20.0, 20.0], 
    'xi' : [0, 1]
}

recovery_init = {
    'loss' : 2.0, 
    'risk_plus' : 1.0, 
    'risk_minus' : 2.0, 
    'init_mean' : 0.0, 
    'init_noise' : 5.0, 
    'theta' : 1.0, 
    'alpha' : 0.5, 
    'rho' : 5.0, 
    'phi' : 10.0, 
    'xi' : 0.5
}

recovery_subjects = 100
recovery_trials = 100


"""
strategy = ['generic UCB', 'generic UCB w seperation', 'relative prospect UCB', 'prospect UCB', 'relative prospect UCB (inner)', 'relative prospect UCB w seperation', 'prospect UCB w seperation', 'relative prospect UCB (inner) w seperation']
st_varaible = {
   'generic UCB' : ['beta', 'mean', 'noise', 'temp'],
   'relative prospect UCB' : ['risk', 'beta', 'loss', 'mean', 'noise', 'temp'],
   'prospect UCB' : ['risk', 'beta', 'loss', 'mean', 'noise', 'temp'],
   'relative prospect UCB (inner)' : ['risk', 'beta', 'loss', 'mean', 'noise', 'temp'],
   'generic UCB w seperation' : ['beta', 'beta2', 'mean', 'noise', 'temp'],
   'relative prospect UCB w seperation' : ['risk', 'beta', 'beta2', 'loss', 'mean', 'noise', 'temp'],
   'prospect UCB w seperation' : ['risk', 'beta', 'beta2', 'loss', 'mean', 'noise', 'temp'],
   'relative prospect UCB (inner) w seperation' : ['risk', 'beta', 'beta2', 'loss', 'mean', 'noise', 'temp']
}

variables = {
    4 : ['beta', 'mean', 'noise', 'temp'],
    5 : ['beta', 'beta2', 'mean', 'noise', 'temp'],
    6 : ['risk', 'beta', 'loss', 'mean', 'noise', 'temp'],
    7 : ['risk', 'beta', 'beta2', 'loss', 'mean', 'noise', 'temp'],
}

n_strategies = len(strategy)"""