1.	Task
We use four-armed bandit task with 300 times of selection. Each bandit faces a decaying random walk as follows:

$\mu_{i,t}^{post} = N(\mu_{i,t}^{pre}, \sigma_o), \quad \mu_{i,t}^{pre}=\lambda\cdot\mu_{i,t}^{post}+(1-\lambda)\cdot\varphi + \nu$ with $\nu\sim N(0,\sigma_D)$.
The reward of bandit i at step t is randomly sampled from $\mu_{i,t}^{pre}$ with standard derivation $\simga_o$. From the reward, next $\mu_{i,t}^{pre}$ is determined with decaying parameter $\lambda$, decaying center $\varphi$ and random walk $\nu$. We use predetermined value from previous studies; $\lambda=0.9836$,$\varphi=50$,$\simga_o=4$, $\sigma_D=2.8$. To check prospect theory, initial bandit’s mean $\mu_{i,t}^{pre}$ is randomly sampled from $N(0,10)$.

In cognitive task structure, state gets predetermined parameters, outputs the bandit’s reward from the response (the index of selected bandit), and updates its environment (bandits’ means).

2. Model
Using Kalman filtering, agents calculate the mean and variance of each bandit from accumulated stimuli. Then the agent randomly selects bandit using prospect theory based UCB-like model and softmax.
In summary, the response is generated as $idx \sim Softmax(\theta \cdot (u(\hat{\mu_{i, t}}^{pre}) + \phi \cdot \sigma_{i, t}^* + \rho \cdot H_{i,t}))$ for calculated mean $\mu$, exploration bonus $\varphi\cdot\sigma^*$, and preservation bonus $\rho\cdot H$.
The detailed version follows:

Learning rule
- $\hat{\mu_{i, t+1}}^{pre} = \lambda_{env} \cdot \hat{\mu_{i, t}}^{post} + (1-\lambda_{env})\cdot \theta_{env}$
- $\hat{\sigma_{i,t+1}}^{2pre} = \lambda_{env}^2 \cdot \hat{\sigma_{i, t}}^{2post}+\sigma_d^2$
	
- $\hat{\mu_{c_t,t}}^{post} = \hat{\mu_{c_t, t}}^{pre}+\kappa_t\delta_t$
- $\hat{\sigma_{c_t,t}}^{2post} = (1-\kappa_t)\hat{\sigma_{c_t,t}}^{2pre}$ where $c_t$ is seleccted bandit
- $\kappa_t = \hat{\sigma_{i, t}}^{2pre} / (\hat{\sigma_{i, t}}^{2pre} + \sigma_o^2)$
- $\delta_t = r_t - \hat{\mu_{c_t,t}}^{pre}$


Decision rule
- $u(x) = \begin{cases}(x- \xi \cdot r_t)^\alpha & \text{if } x \geq \xi \cdot r_t \\ -\lambda \cdot (- x + \xi \cdot r_t)^\beta & \text{if } x < \xi \cdot r_t \end{cases}$
- $Softmax(\theta \cdot (u(\hat{\mu_{i, t}}^{pre}) + \phi \cdot \sigma_{i, t}^* + \rho \cdot H_{i,t}))$

- $H_{i, t} = H_{i, t-1} + \alpha_H \cdot (I_{i, t-1} - H_{i, t-1})$
- $\sigma_{i,t}^* = \sqrt{Var[u(x)]} = \sqrt{\int_{-\infty}^\infty \frac{1}{\hat{\sigma_{i,t}}\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\hat{\mu_{i,t}}}{\hat{\sigma_{i,t}}})^2} \cdot [u(x) - \mathbb{E}[u(x)]]^2 dx} = \sqrt{\Phi_{2\alpha}(X) + \lambda^2 \cdot \Phi_{2\beta}(X) - (\Phi_\alpha(X) - \lambda \cdot \Phi_\beta (X))^2}$
    - $\Phi_r(x) := \frac{\sigma^r}{\sqrt{2\pi}} e^{-\frac{1}{4}x^2}\Gamma(r+1)D_{-r-1}(-x)$
	- $X := (\hat{\mu_{i, t}}^{pre} - \xi \cdot r_t) / \hat{\sigma_{i, t}}^{pre}$ 