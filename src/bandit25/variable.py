from dataclasses import dataclass

import torch

@dataclass
class Env:
    env: torch.Tensor       # structure (horizon x structure if restless)
    stationary : int        # 1 -> stationary, 0 -> restless
    structure: tuple = None # structure of bandit in each env
    horizon: int = None     # number of horizon in each task
    mean: float = None        # mean of bandits & decay center (restless)
    noise: float = None       # noise of bandits (stationary) & obsevational noise of bandits (restless)
    noise_d : float = None    # step noise of bandits (restless)
    lamb : float = None     # decay parameter (restless)
    l : float = None        # correlated value (correlated)
    name: str = None

variance = 10
noise = 10
horizon = 70
num = 100
mean = [100, 0, -100]
env_name = ['pos', 'mix', 'neg']

# risk and theta are kept small enough that simulated choices stay stochastic: with risk up to 3 and
# theta up to 5, 99% of agents chose almost deterministically (mean p(chosen) > 0.9) and the likelihood
# carried little information. In these ranges p(chosen) spreads over 0.30 / 0.65 / 0.99 (10/50/90%).
recovery_range = {
    'loss' : [0.0, 10.0],
    'risk_plus' : [0.2, 1.2],
    'risk_minus' : [0.2, 1.2],
    'init_mean' : [-100.0, 100.0],
    'init_noise' : [2.0, 10.0],
    'theta' : [1e-3, 0.2],
    'alpha' : [0.0, 1.0],
    'rho' : [-20.0, 20.0],
    'phi' : [-20.0, 20.0],
    'xi' : [0, 1]
}

recovery_init = {
    'loss' : 2.0,
    'risk_plus' : 0.7,
    'risk_minus' : 0.7,
    'init_mean' : 0.0,
    'init_noise' : 5.0,
    'theta' : 0.1,
    'alpha' : 0.5,
    'rho' : 5.0,
    'phi' : 10.0,
    'xi' : 0.5
}
