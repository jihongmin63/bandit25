# theta, alpha (preservation calculation), init_noise, xi (anchoring effect)

# 격자로 만들어서 멀어질수록 loss가 커지는지 확인
import torch
from multiprocessing import Pool, set_start_method, cpu_count
from scipy.optimize import minimize
from tqdm import tqdm

import pickle

from bandit25.variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from bandit25.vectorized_function import vectorized_simulate
from bandit25.vectorized_models import Vectorized_Kalman_SM, free_varaibles
import numpy as np


from bandit25.function import simulate
from bandit25.models import Kalman_SM, free_varaibles

from torchviz import make_dot

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

with open("data/envs/restless_envs", 'rb') as f:
    restless_envs = pickle.load(f)
    
envs = restless_envs[1]

num = 30
selected_envs = [envs[t] for t in range(num)]

problematic_params = ['theta']
for problematic_param in problematic_params:
    print(f"<{problematic_param}>")
    res = []

    values = torch.linspace(
        recovery_range[problematic_param][0],
        recovery_range[problematic_param][1],
        50
    )
    X, Y = torch.meshgrid(values, values, indexing='xy')
    pairs = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # --- 1️⃣ 먼저 구조 고정 및 index 찾기 ---
    var_names = free_varaibles['Kalman_SM']
    idx = var_names.index(problematic_param)

    # 초기 구조 샘플 만들기
    src_params_base, tgt_params_base = [], []
    for _ in range(num):
        src_param, tgt_param = [], []
        for var_name in var_names:
            if var_name == problematic_param:
                src_param.append(0.0)  # 자리만 잡기
                tgt_param.append(0.0)
            else:
                _val = (
                    torch.rand(1)
                    * (recovery_range[var_name][1] - recovery_range[var_name][0])
                    + recovery_range[var_name][0]
                ).item()
                src_param.append(_val)
                tgt_param.append(_val)
        src_params_base.append(src_param)
        tgt_params_base.append(tgt_param)

    # torch tensor로 고정
    src_params = torch.tensor(src_params_base, dtype=torch.float64)
    tgt_params = torch.tensor(tgt_params_base, dtype=torch.float64)
    src_params = src_params.transpose(0, 1).unsqueeze(-1)
    tgt_params = tgt_params.transpose(0, 1).unsqueeze(-1)

    # --- 2️⃣ pair별로 problematic_param만 교체 ---
    for src, tgt in tqdm(pairs, desc=f"{problematic_param} pairs", total=len(pairs)):
        src_params[idx, :] = src.item()
        tgt_params[idx, :] = tgt.item()

        al, rl, _, _ = vectorized_simulate(
            src_params, [envs[t] for t in range(num)], Vectorized_Kalman_SM
        )
        _, _, _, _, prob = vectorized_simulate(
            tgt_params, selected_envs, Vectorized_Kalman_SM,
            test=1, real_actions=al, real_rewards=rl
        )
        res.append(prob.t().mean())

    # --- 3️⃣ 결과 저장 ---
    with open(f"results/params/Loss_sanity_{problematic_param}", "wb") as f:
        pickle.dump(res, f)