import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
from variable import recovery_range
import pickle

problematic_params = ['theta', 'alpha', 'init_noise', 'init_mean', 'xi']
for problematic_param in problematic_params:
        
    with open(f"params/Loss_sanity_{problematic_param}", 'rb') as f:
        res = pickle.load(f)
        res_tensor = torch.stack([r.flatten() for r in res])  # shape: (2500,)

    # 50x50 grid로 reshape
    Z = res_tensor.reshape(30, 30).numpy()

    # --- grid 생성 (X, Y는 위에서 만든 values 사용)
    values = torch.linspace(
        recovery_range[problematic_param][0],
        recovery_range[problematic_param][1],
        30,
        dtype=torch.float64
    )
    X, Y = torch.meshgrid(values, values, indexing='xy')
    X = X.numpy()
    Y = Y.numpy()

    # --- 3D 시각화 ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(f'{problematic_param} src')
    ax.set_ylabel(f'{problematic_param} tgt')
    ax.set_zlabel('Loss or Prob')
    ax.set_title(f'Loss landscape for {problematic_param}')

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    plt.show()
    plt.savefig(f"images/loss_sanity_{problematic_param}.png", dpi=300)
    plt.close()