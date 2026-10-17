import numpy as np
import pytest
import torch

from bandit25 import models, vectorized_models
from bandit25.function import prospect
from bandit25.vectorized_function import prospect as vectorized_prospect

f64 = dict(dtype=torch.float64)
MODULES = [models, vectorized_models]


def variance(module, a, b, lam, mu, sigma, xi=0.0, reward=0.0):
    """Evaluate module.Variance for one subject, in the tensor layout that module expects."""
    t = lambda z: torch.as_tensor(z, **f64)
    if module is vectorized_models:
        p = lambda z: t(z).reshape(1, 1)
        return module.Variance(p(a), p(b), p(lam), t(mu).reshape(1, -1), t(sigma).reshape(1, -1),
                               p(xi), p(reward)).reshape(-1)
    return module.Variance(t(a), t(b), t(lam), t(mu), t(sigma), t(xi), t(reward))


def test_parabolic_cylinder_gradcheck_vectorized():
    torch.manual_seed(0)
    v = (-1 - 4 * torch.rand(3, 1, **f64)).requires_grad_()
    x = (6 * torch.rand(3, 4, **f64) - 3).requires_grad_()
    assert torch.autograd.gradcheck(vectorized_models.Parabolic_Cylinder.apply, (v, x),
                                    eps=1e-6, atol=1e-5, rtol=1e-3)


def test_parabolic_cylinder_gradcheck_single():
    torch.manual_seed(0)
    v = torch.tensor(-2.3, **f64, requires_grad=True)
    x = (6 * torch.rand(4, **f64) - 3).requires_grad_()
    assert torch.autograd.gradcheck(models.Parabolic_Cylinder.apply, (v, x),
                                    eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("a, b, lam, mu, sigma, xi, reward", [
    (0.8, 0.9, 2.0, 5.0, 4.0, 0.0, 0.0),
    (0.5, 1.5, 3.0, -10.0, 6.0, 0.0, 0.0),
    (2.0, 2.5, 1.0, 20.0, 15.0, 0.5, 12.0),
])
def test_variance_matches_monte_carlo(module, a, b, lam, mu, sigma, xi, reward):
    x = np.random.default_rng(0).normal(mu, sigma, 2_000_000) - xi * reward
    u = np.clip(x, 0, None) ** a - lam * np.clip(-x, 0, None) ** b
    got = variance(module, a, b, lam, [mu], [sigma], xi, reward).item()
    assert got == pytest.approx(u.var(), rel=0.02)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("a", [0.05, 0.5, 1.0, 3.0])
def test_variance_far_from_reference_is_finite_and_continuous(module, a):
    sigma = 2.0
    cs = np.array([-200.0, -60.0, -40.0001, -39.9999, 39.9999, 40.0001, 60.0, 200.0])
    risk = torch.tensor(a, **f64, requires_grad=True)
    var = variance(module, risk, risk, 2.0, cs * sigma, np.full_like(cs, sigma))
    assert torch.isfinite(var).all()
    var.sum().backward()
    assert torch.isfinite(risk.grad)
    # the closed form and the delta-method branch agree where they meet
    assert var[2].item() == pytest.approx(var[3].item(), rel=5e-3)
    assert var[4].item() == pytest.approx(var[5].item(), rel=5e-3)


@pytest.mark.parametrize("module", MODULES)
def test_variance_gradient_matches_finite_difference(module):
    mu, sigma = [5.0, -3.0, 10.0, 0.5], [4.0, 4.0, 15.0, 3.0]
    theta0 = [0.8, 0.9, 2.0, 0.3]  # risk_plus, risk_minus, loss, xi
    params = [torch.tensor(p, **f64, requires_grad=True) for p in theta0]
    var = variance(module, *params[:3], mu, sigma, params[3], 2.0)
    assert var.requires_grad
    var.sum().backward()

    h = 1e-5
    for i, p in enumerate(params):
        up, dn = list(theta0), list(theta0)
        up[i] += h
        dn[i] -= h
        num = (variance(module, *up[:3], mu, sigma, up[3], 2.0).sum()
               - variance(module, *dn[:3], mu, sigma, dn[3], 2.0).sum()) / (2 * h)
        assert p.grad.item() == pytest.approx(num.item(), rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("fn", [prospect, vectorized_prospect])
@pytest.mark.parametrize("r", [1.0, 0.3, 0.05, 0.01])
def test_prospect_keeps_gain_and_loss_domains_apart(fn, r):
    x = torch.tensor([12.0, -8.0, 2.0], **f64)
    r_ = torch.tensor(r, **f64)
    # reference point xi * prev_result = 2
    u = fn(x, torch.tensor(5.0, **f64), r_, r_, xi=torch.tensor(0.5, **f64), prev_result=torch.tensor(4.0, **f64))
    expected = torch.tensor([10.0 ** r, -5.0 * 10.0 ** r, 0.0], **f64)
    assert torch.allclose(u, expected)
