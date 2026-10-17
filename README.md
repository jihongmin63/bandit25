## 1.	Task
We use four-armed bandit task with 300 times of selection. Each bandit faces a decaying random walk as follows:

$\mu_{i,t+1}=\lambda\cdot\mu_{i,t}+(1-\lambda)\cdot\varphi + \nu$ with $\nu\sim N(0,\sigma_D)$, and the reward $r_t \sim N(\mu_{c_t,t}, \sigma_o)$.
The reward of the chosen bandit is sampled around its current mean with standard deviation $\sigma_o$, and every mean decays toward the center $\varphi$ with decaying parameter $\lambda$ plus the random walk $\nu$ (Daw et al., 2006), which is exactly what the Kalman filter below assumes. We use predetermined value from previous studies; $\lambda=0.9836,\sigma_o=4, \sigma_D=2.8$. To check prospect theory, the centers are $\varphi = 100, 0, -100$ (`pos`, `mix`, `neg`) and the initial means are sampled from $N(\varphi,10)$; the recovery scripts use the `mix` environments.

<img width="1200" height="500" alt="Image" src="https://github.com/user-attachments/assets/9765887e-f2c7-43b2-9551-fe909464f7ec" />

In cognitive task structure, state gets predetermined parameters, outputs the bandit’s reward from the response (the index of selected bandit), and updates its environment (bandits’ means).

## 2. Model
Using Kalman filtering, agents calculate the mean and variance of each bandit from accumulated stimuli. Then the agent randomly selects bandit using prospect theory based UCB-like model and softmax.
In summary, the response is generated as $idx \sim Softmax(\theta \cdot (u(\hat{\mu_{i, t}}^{pre}) + \phi \cdot \sigma_{i, t}^* + \rho \cdot H_{i,t}))$ for calculated mean $\mu$, exploration bonus $\varphi\cdot\sigma^*$, and preservation bonus $\rho\cdot H$.
The detailed version follows:

**Learning rule**
- $\hat{\mu_{i, t+1}}^{pre} = \lambda_{env} \cdot \hat{\mu_{i, t}}^{post} + (1-\lambda_{env})\cdot \theta_{env}$
- $\hat{\sigma_{i,t+1}}^{2pre} = \lambda_{env}^2 \cdot \hat{\sigma_{i, t}}^{2post}+\sigma_d^2$
	
- $\hat{\mu_{c_t,t}}^{post} = \hat{\mu_{c_t, t}}^{pre}+\kappa_t\delta_t$
- $\hat{\sigma_{c_t,t}}^{2post} = (1-\kappa_t)\hat{\sigma_{c_t,t}}^{2pre}$ where $c_t$ is seleccted bandit
- $\kappa_t = \hat{\sigma_{i, t}}^{2pre} / (\hat{\sigma_{i, t}}^{2pre} + \sigma_o^2)$
- $\delta_t = r_t - \hat{\mu_{c_t,t}}^{pre}$


**Decision rule**
- $u(x) = \text{max}(x- \xi \cdot r_t, 0)^\alpha -\lambda \cdot \text{max}(- x + \xi \cdot r_t, 0)^\beta$
- $Softmax(\theta \cdot (u(\hat{\mu_{i, t}}^{pre}) + \phi \cdot \sigma_{i, t}^* + \rho \cdot H_{i,t}))$

- $H_{i, t} = H_{i, t-1} + \alpha_H \cdot (I_{i, t-1} - H_{i, t-1})$
- $\sigma_{i,t}^* = \sqrt{Var[u(x)]} = \sqrt{\int_{-\infty}^\infty \frac{1}{\hat{\sigma_{i,t}}\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\hat{\mu_{i,t}}}{\hat{\sigma_{i,t}}})^2} \cdot [u(x) - \mathbb{E}[u(x)]]^2 dx} = \sqrt{\Phi_{2\alpha}(X) + \lambda^2 \cdot \Phi_{2\beta}(X) - (\Phi_\alpha(X) - \lambda \cdot \Phi_\beta (X))^2}$
    - $\Phi_r(x) := \frac{\sigma^r}{\sqrt{2\pi}} e^{-\frac{1}{4}x^2}\Gamma(r+1)D_{-r-1}(-x)$
	- $X := (\hat{\mu_{i, t}}^{pre} - \xi \cdot r_t) / \hat{\sigma_{i, t}}^{pre}$ 

**Free parameters** (`free_variables['Kalman_SM']`, bounds in `recovery_range`)

| Code name | Symbol | Meaning | Range |
|---|---|---|---|
| `loss` | $\lambda$ | loss aversion | [0, 10] |
| `risk_plus` | $\alpha$ | curvature of gains | [0.2, 1.2] |
| `risk_minus` | $\beta$ | curvature of losses | [0.2, 1.2] |
| `init_mean` | $\hat{\mu}_{i,0}$ | initial estimate of every arm | [-100, 100] |
| `init_noise` | $\hat{\sigma}_{i,0}$ | initial uncertainty of every arm | [2, 10] |
| `theta` | $\theta$ | inverse temperature | [1e-3, 0.2] |
| `alpha` | $\alpha_H$ | learning rate of the preservation trace $H$ | [0, 1] |
| `rho` | $\rho$ | weight of the preservation bonus | [-10, 10] |
| `phi` | $\phi$ | weight of the exploration bonus $\sigma^*$ | [-10, 10] |
| `xi` | $\xi$ | weight of the previous reward in the reference point | [0, 1] |

$\lambda_{env}$, $\theta_{env}$ (the decay center, $\varphi$ in §1), $\sigma_o$ and $\sigma_d$ are environment constants, not free parameters.

## 3. Project structure

| Path | Contents |
|---|---|
| `src/bandit25/` | Core package: environments, models, simulation, fitting |
| `scripts/make_envs.py` | Generates `data/envs/*` and the sample env figures |
| `scripts/recovery/` | Parameter recovery experiments and their plots |
| `scripts/loss_sanity/` | Loss sanity checks for individual parameters |
| `notebooks/` | Exploratory notebooks |
| `results/figures/`, `results/params/` | Generated figures and pickled recovery results |
| `tests/` | Checks of the numerics and the fitting (`pytest`) |
| `data/`, `logs/` | Generated environments and logs (not tracked) |

Install once with `pip install -e .`, then run scripts from the repository root:

```bash
python scripts/make_envs.py                               # data/envs/ (seed 42)
python scripts/recovery/vectorized_parameter_recovery.py  # needs data/envs/restless_envs
python scripts/recovery/vectorized_visualize.py
python scripts/loss_sanity/loss_sanity.py theta           # any free parameter; --grid 50 by default
pytest
```

`python scripts/make_envs.py --process original` writes `data/envs/restless_envs_original` instead: the environments
used before c165b65 (same draws as the original envs.py), where the drift also takes the observation noise, so its
step SD is about 4.8 rather than $\sigma_D = 2.8$.
