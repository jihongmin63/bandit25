import torch

from bandit25.variable import Env
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_variables
from bandit25.fitting import initial_params


def test_switch_rate_counts_changes_of_arm():
    horizon = 5
    envs = [Env(env=torch.zeros(horizon, 4), stationary=0, structure=(4,), horizon=horizon,
                mean=0.0, noise=4.0, noise_d=2.8, lamb=0.9836) for _ in range(2)]
    arms = torch.tensor([[0, 0], [1, 0], [1, 0], [2, 0], [3, 0]])  # subject 0 switches 3 times, subject 1 never
    actions = [torch.nn.functional.one_hot(a, num_classes=4).float() for a in arms]
    rewards = [torch.zeros(2, 1) for _ in range(horizon)]

    _, _, _, switch, _ = vectorized_simulate(initial_params(free_variables['Kalman_SM'], 2), envs, Vectorized_Kalman_SM,
                                             test=1, real_actions=actions, real_rewards=rewards)
    assert torch.allclose(switch, torch.tensor([3 / 4, 0.0]))
