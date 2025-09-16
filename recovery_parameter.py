import torch
from tqdm import tqdm
import pickle
import numpy as np

from variable import recovery_range, recovery_subjects, recovery_trials, recovery_init, Env
from function import simulate
from models import Kalman_SM, free_varaibles

np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.autograd.set_detect_anomaly(True)

def recover_subject(subject_idx, envs, lr=1e-2, step=500):
    param = []
    bound = []
    x0 = []
    for var_name in free_varaibles['Kalman_SM']:
        param_val = (torch.rand(1) * (recovery_range[var_name][1] - recovery_range[var_name][0]) + recovery_range[var_name][0]).item()
        param.append(param_val)
        bound.append([recovery_range[var_name][0], recovery_range[var_name][1]])
        x0.append(recovery_init[var_name])
    param = torch.tensor(param, dtype=torch.float64)
    
    recovered_x0 = []
    # trial loop tqdm
    for i in tqdm(range(recovery_trials), desc=f"Subject {subject_idx}", leave=False):
        x = torch.nn.Parameter(torch.tensor(x0, requires_grad=True, dtype=torch.float64))
        optimizer = torch.optim.Adam([x], lr=lr)
        al, rl, _, _ = simulate(param, envs[i], Kalman_SM)
        
        # step loop tqdm
        for time in tqdm(range(step), desc=f"Trial {i}", leave=False):
            optimizer.zero_grad()
            _, _, _, _, prob = simulate(x, envs[i], Kalman_SM, test=1, real_actions=al, real_rewards=rl)
            loss = -prob.sum()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            with torch.no_grad():
                for j, p in enumerate(x):
                    p.clamp_(min=bound[j][0], max=bound[j][1])
        
        recovered_x0.append(x.detach())
    
    return [param, recovered_x0]

if __name__ == "__main__":
    with open("envs/restless_envs", 'rb') as f:
        restless_envs = pickle.load(f)
    
    envs = restless_envs[1]

    recovered_params_list = []
    # subject loop tqdm
    for idx in tqdm(range(recovery_subjects), desc="Subjects"):
        result = recover_subject(idx, envs)
        recovered_params_list.append(result)

    with open("params/restless_params", 'wb') as f:
        pickle.dump(recovered_params_list, f)
