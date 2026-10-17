import pandas as pd
import pytest
import torch

from bandit25.data import load_choices, save_choices, task_envs
from bandit25.fitting import sample_params
from bandit25.variable import Env
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_variables

names = free_variables['Kalman_SM']


def simulated(n=3, horizon=20):
    torch.manual_seed(0)
    envs = [Env(env=torch.randn(horizon, 4) * 10, stationary=0, structure=(4,), horizon=horizon,
                mean=0.0, noise=4.0, noise_d=2.8, lamb=0.9836) for _ in range(n)]
    params = sample_params(names, n)
    al, rl, _, _ = vectorized_simulate(params, envs, Vectorized_Kalman_SM)
    return params, al, rl


def test_round_trip_gives_the_same_likelihood(tmp_path):
    params, al, rl = simulated()
    path = tmp_path / "choices.csv"
    save_choices(path, al, rl, subjects=["s1", "s2", "s3"])

    subjects, actions, rewards = load_choices(path)
    assert subjects == ["s1", "s2", "s3"]
    assert len(actions) == len(al) and actions[0].shape == (3, 4) and rewards[0].shape == (3, 1)

    envs = task_envs(3, len(al), center=0.0)
    *_, expected = vectorized_simulate(params, envs, Vectorized_Kalman_SM, test=1, real_actions=al, real_rewards=rl)
    *_, got = vectorized_simulate(params, envs, Vectorized_Kalman_SM, test=1, real_actions=actions, real_rewards=rewards)
    assert torch.allclose(got, expected, rtol=1e-5)


def test_rows_may_come_in_any_order(tmp_path):
    _, al, rl = simulated()
    path = tmp_path / "choices.csv"
    save_choices(path, al, rl)
    _, actions, rewards = load_choices(path)

    shuffled = tmp_path / "shuffled.csv"
    pd.read_csv(path, float_precision="round_trip").sample(frac=1, random_state=0).to_csv(shuffled, index=False)
    _, actions2, rewards2 = load_choices(shuffled)
    assert all(torch.equal(a, b) for a, b in zip(actions, actions2))
    assert all(torch.equal(a, b) for a, b in zip(rewards, rewards2))


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns="reward"), "missing columns"),
    (lambda df: df.drop(index=5), "same number of trials"),
    (lambda df: df.assign(choice=df["choice"].where(df.index != 3, 4)), "choice must be"),
    (lambda df: df.assign(reward=df["reward"].where(df.index != 3, None)), "empty values"),
])
def test_invalid_files_are_rejected(tmp_path, change, message):
    _, al, rl = simulated()
    path = tmp_path / "choices.csv"
    save_choices(path, al, rl)
    change(pd.read_csv(path)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_choices(path)
