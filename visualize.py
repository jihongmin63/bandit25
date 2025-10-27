import pickle
import pandas as pd
from models import free_varaibles
import seaborn as sns 
import numpy as np
import torch
import matplotlib.pyplot as plt
with open("params/Maksed_Restless_params","rb") as f:
    original_data, recovered_data = pickle.load(f)
    

fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 가로 5, 세로 2
axes = axes.flatten()  # 2차원을 1차원 리스트로 펼침

for idx, var in enumerate(free_varaibles['Kalman_SM']):
    print(idx, var)
    ax = axes[idx]

    data = []
    orig_col, rec_col = f"original {var}", f"recovered {var}"

    for i in range(len(recovered_data)):
        data.append([original_data[i][idx], recovered_data[i][idx][0]])

    df = pd.DataFrame(data, columns=[orig_col, rec_col])

    try:
        sns.kdeplot(data=df, x=orig_col, y=rec_col, fill=True, levels=10, ax=ax, warn_singular=False)
    except Exception as e:
        sns.scatterplot(data=df, x=orig_col, y=rec_col, ax=ax, alpha=0.6)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    r = np.corrcoef(df[orig_col], df[rec_col])[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

    ax.set_title(f"{var}")

plt.tight_layout()
plt.savefig("images/recovery/Masked_all_parameters.png", dpi=300)
plt.close()
