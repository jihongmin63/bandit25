import pandas as pd
import torch

from bandit25.variable import Env
from bandit25.envs import lamb, noise_o, noise_d

COLUMNS = ["subject", "trial", "choice", "reward"]


def load_choices(path, arms=4):
    """
        Choices of every subject from a long-format CSV, one row per trial

        Columns:
            subject : any id
            trial : order of trials within a subject (sorted, need not start at 0 or 1)
            choice : chosen arm, 0 ~ arms-1
            reward : reward received on that trial

        Every subject must have the same number of trials and a choice on every trial.

        Returns:
            subjects (list) : subject ids, in the order of the batch
            actions (list of tensors, (n_subjects, arms)) : one-hot choices, one tensor per trial
            rewards (list of tensors, (n_subjects, 1)) : rewards, one tensor per trial
    """
    df = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected {COLUMNS}")
    if df[COLUMNS].isna().any().any():
        rows = df.index[df[COLUMNS].isna().any(axis=1)].tolist()
        raise ValueError(f"{path}: empty values in rows {rows[:10]}; drop or fill missed trials first")
    if not df["choice"].isin(range(arms)).all():
        bad = sorted(set(df["choice"]) - set(range(arms)))
        raise ValueError(f"{path}: choice must be 0 ~ {arms - 1}, got {bad[:10]}")
    if df.duplicated(["subject", "trial"]).any():
        raise ValueError(f"{path}: some subjects have the same trial number twice")

    df = df.sort_values(["subject", "trial"], kind="stable")
    counts = df.groupby("subject", sort=False).size()
    if counts.nunique() != 1:
        raise ValueError(f"{path}: every subject needs the same number of trials, got {counts.to_dict()}")

    subjects = counts.index.tolist()
    n, horizon = len(subjects), int(counts.iloc[0])
    choice = torch.tensor(df["choice"].to_numpy(dtype="int64").reshape(n, horizon))
    reward = torch.tensor(df["reward"].to_numpy(dtype="float64").reshape(n, horizon))

    actions = [torch.nn.functional.one_hot(choice[:, t], num_classes=arms).double() for t in range(horizon)]
    rewards = [reward[:, t:t + 1] for t in range(horizon)]
    return subjects, actions, rewards


def save_choices(path, actions, rewards, subjects=None):
    """
        Write actions and rewards (as returned by vectorized_simulate) in the format of load_choices
    """
    choice = torch.stack(actions).argmax(dim=-1)            # (horizon, n)
    reward = torch.stack(rewards)[..., 0].detach()          # (horizon, n)
    horizon, n = choice.shape
    subjects = list(range(n)) if subjects is None else list(subjects)
    df = pd.DataFrame({
        "subject": [s for s in subjects for _ in range(horizon)],
        "trial": list(range(1, horizon + 1)) * n,
        "choice": choice.T.reshape(-1).tolist(),
        "reward": reward.T.reshape(-1).tolist(),
    })
    df.to_csv(path, index=False)


def task_envs(n, horizon, center, arms=4, noise=noise_o, noise_d=noise_d, lamb=lamb):
    """
        Envs holding only the task constants the model uses, for fitting observed choices

        Args:
            n, horizon (int) : number of subjects and trials
            center (float) : decay center of the arm means (theta_env)
            noise, noise_d, lamb (float) : observation noise, drift noise and decay of the task
    """
    return [Env(env=None, stationary=0, structure=(arms,), horizon=horizon, mean=center,
                noise=noise, noise_d=noise_d, lamb=lamb, name="task") for _ in range(n)]
