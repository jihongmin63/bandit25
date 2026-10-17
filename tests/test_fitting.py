import torch

from bandit25.variable import Env
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_variables
from bandit25.fitting import bounds, fit, initial_params, sample_params, to_constrained, to_unconstrained

names = free_variables['Kalman_SM']


def short_envs(n, horizon=40):
    torch.manual_seed(0)
    envs = []
    for _ in range(n):
        mu = torch.normal(0.0, 10.0, size=(4,))
        rows = [mu]
        for _ in range(horizon - 1):
            mu = 0.9836 * mu + torch.normal(0.0, 2.8, size=(4,))
            rows.append(mu)
        envs.append(Env(env=torch.stack(rows), stationary=0, structure=(4,), horizon=horizon,
                        mean=0.0, noise=4.0, noise_d=2.8, lamb=0.9836))
    return envs


def test_unconstrained_round_trip():
    lo, hi = bounds(names)
    x = sample_params(names, 5)
    assert torch.allclose(to_constrained(to_unconstrained(x, lo, hi), lo, hi), x, atol=1e-6)


def test_fit_improves_likelihood_within_bounds():
    envs = short_envs(3)
    true = sample_params(names, 3)
    al, rl, _, _ = vectorized_simulate(true, envs, Vectorized_Kalman_SM)
    x0 = initial_params(names, 3)
    *_, ll0 = vectorized_simulate(x0, envs, Vectorized_Kalman_SM, test=1, real_actions=al, real_rewards=rl)

    x, ll = fit(x0, names, envs, al, rl, max_eval=15)
    lo, hi = bounds(names)
    assert ((x >= lo) & (x <= hi)).all()
    assert (ll >= ll0).all() and (ll > ll0).any()


def test_fit_keeps_fixed_parameters():
    envs = short_envs(2)
    true = sample_params(names, 2)
    al, rl, _, _ = vectorized_simulate(true, envs, Vectorized_Kalman_SM)
    x0 = true.clone()
    x0[names.index('phi')] = 0.0

    x, _ = fit(x0, names, envs, al, rl, max_eval=10, free=['phi'])
    fixed = [i for i, v in enumerate(names) if v != 'phi']
    assert torch.equal(x[fixed], x0[fixed])
