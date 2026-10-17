import torch
from tqdm import tqdm

from bandit25.variable import recovery_range, recovery_init
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM


def bounds(names, dtype=torch.float64):
    """
        Lower and upper bounds of recovery_range, shaped (n_params, 1, 1) to broadcast
        against parameters of shape (n_params, n_subjects, 1)
    """
    lo = torch.tensor([recovery_range[v][0] for v in names], dtype=dtype).view(-1, 1, 1)
    hi = torch.tensor([recovery_range[v][1] for v in names], dtype=dtype).view(-1, 1, 1)
    return lo, hi


def sample_params(names, n, dtype=torch.float64):
    """
        Uniform samples inside recovery_range

        Returns:
            (tensor, (n_params, n, 1))
    """
    lo, hi = bounds(names, dtype)
    return lo + (hi - lo) * torch.rand(len(names), n, 1, dtype=dtype)


def initial_params(names, n, dtype=torch.float64):
    """
        recovery_init for n subjects

        Returns:
            (tensor, (n_params, n, 1))
    """
    x0 = torch.tensor([recovery_init[v] for v in names], dtype=dtype)
    return x0.view(-1, 1, 1).expand(-1, n, 1).clone()


def to_unconstrained(x, lo, hi, eps=1e-6):
    return torch.logit(((x - lo) / (hi - lo)).clamp(eps, 1 - eps))


def to_constrained(z, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(z)


def fit(x0, names, envs, actions, rewards, model=Vectorized_Kalman_SM, max_eval=200, free=None, desc=None):
    """
        Maximum-likelihood fit of all subjects at once

        Parameters are optimized as z with x = lo + (hi - lo) * sigmoid(z), so the bounds hold
        and every parameter is on the same scale. L-BFGS with a line search reaches the
        log-likelihood of the true parameters in ~30 evaluations, where Adam needs hundreds.

        Args:
            x0 (tensor, (n_params, n_subjects, 1)) : starting points inside recovery_range
            names (list of str) : parameter names in the order of x0
            envs (list of Env) : env of each subject
            actions, rewards (list) : observed data, as returned by vectorized_simulate
            max_eval (int) : maximum number of likelihood evaluations
            free (list of str) : parameters to fit; the others stay at x0 (masked recovery)
            desc (str) : tqdm description; no progress bar if None

        Returns:
            x (tensor, (n_params, n_subjects, 1)) : best parameters seen for each subject
            log_lik (tensor, (n_subjects,)) : log-likelihood at x
    """
    lo, hi = bounds(names, x0.dtype)
    fixed = torch.zeros(len(names), 1, 1, dtype=torch.bool)
    if free is not None:
        fixed = torch.tensor([v not in free for v in names]).view(-1, 1, 1)

    z = torch.nn.Parameter(to_unconstrained(x0, lo, hi))
    optimizer = torch.optim.LBFGS([z], max_iter=max_eval, max_eval=max_eval, line_search_fn="strong_wolfe")

    best_x = x0.detach().clone()
    best_log_lik = torch.full((x0.size(1),), -float('inf'), dtype=x0.dtype)
    progress = tqdm(total=max_eval, desc=desc, leave=False, disable=desc is None)

    def closure():
        optimizer.zero_grad()
        x = torch.where(fixed, x0, to_constrained(z, lo, hi))
        *_, log_lik = vectorized_simulate(x, envs, model, test=1, real_actions=actions, real_rewards=rewards)

        # the summed objective can trade one subject for another, so keep each subject's best point
        with torch.no_grad():
            better = log_lik > best_log_lik
            best_x[:, better] = x[:, better]
            best_log_lik[better] = log_lik[better]

        # subjects are independent, so one backward gives each subject its own gradient
        loss = (-log_lik).sum()
        loss.backward()
        progress.update()
        return loss

    optimizer.step(closure)
    progress.close()
    return best_x, best_log_lik
