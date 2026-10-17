
import torch
from bandit25.variable import horizon, Env
from bandit25.vectorized_function import prospect, vectorized_simulate
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import mvn
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy import special
import pickle
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import numpy as np
from scipy import special

class Parabolic_Cylinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        v: (n,1) tensor
        x: (n, m) tensor
        returns: (n,m) tensor
        """
        v_np = v.detach().cpu().numpy()  # (n,1)
        x_np = x.detach().cpu().numpy()  # (n,m)

        # pbdv는 ufunc라서 (n,1)과 (n,m)을 broadcasting해 한 번에 계산
        out_np, _ = special.pbdv(v_np, x_np)
        out = torch.from_numpy(out_np).to(x.device, dtype=x.dtype)

        ctx.save_for_backward(v, x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, x = ctx.saved_tensors
        v_np = v.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()

        eps = 1e-6

        # grad x
        _, grad_x_np = special.pbdv(v_np, x_np)

        # grad v via finite difference, element-wise (n,m)
        out_plus, _ = special.pbdv(v_np + eps, x_np)
        out_minus, _ = special.pbdv(v_np - eps, x_np)
        grad_v_np = (out_plus - out_minus) / (2 * eps)

        grad_x = torch.from_numpy(grad_x_np).to(x.device, dtype=x.dtype) * grad_output
        # v is shared by the m arms of each row, so their contributions are summed
        grad_v = torch.from_numpy(grad_v_np).to(v.device, dtype=v.dtype) * grad_output
        grad_v = grad_v.sum_to_size(v.shape)

        return grad_v, grad_x


def Variance(risk_plus, risk_minus, loss, mu, sigma, xi, reward, c_max = 40.0):
    # reward: (batch, 1) or (batch,)
    reward = reward.unsqueeze(-1) if reward.dim() == 1 else reward
    d = mu - xi * reward
    c = d / sigma  # broadcasting 지원

    # |c| > ~53 overflows exp/pbdv into nan, so far arms use the delta method below
    far = c.abs() > c_max
    c = torch.clamp(c, -c_max, c_max)

    def Helper(r, c):
        """
        Helper function calculating distorted sigma for 2D batch
        """
        v1 = torch.pow(sigma, r)
        v2 = torch.sqrt(torch.tensor(2 * np.pi, device=c.device, dtype=c.dtype))
        v3 = torch.exp(-(c*c)/4)
        v4 = torch.exp(torch.lgamma(r+1))
        v5 = Parabolic_Cylinder.apply(-r-1, -c)
        return v1 * v3 * v4 * v5 / v2

    var = Helper(2 * risk_plus, c) \
          + loss*loss * Helper(2 * risk_minus, -c) \
          - torch.square(Helper(risk_plus, c) - loss*Helper(risk_minus, -c))

    # almost all mass on one side of the reference: Var[u(X)] ~ (u'(d) * sigma)^2
    d_far = torch.clamp(d.abs(), min=c_max * sigma)
    slope = torch.where(d > 0, risk_plus * torch.pow(d_far, risk_plus - 1),
                        loss * risk_minus * torch.pow(d_far, risk_minus - 1))
    var = torch.where(far, torch.square(slope * sigma), var)

    # clamp without no_grad so the gradient through sigma* is kept
    return torch.clamp(var, min=1e-10)


free_varaibles = {
    'Kalman_SM' : ['loss', 'risk_plus', 'risk_minus', 'init_mean', 'init_noise', 'theta', 'alpha', 'rho', 'phi', 'xi']
}

class Vectorized_Kalman_SM :
    def __init__(self, params, envs) :
        self.loss, self.risk_plus, self.risk_minus, self.init_mean, self.init_noise, self.theta, self.alpha, self.rho, self.phi, self.xi = params
        self.n = params.size(dim=1)
        self.structure = envs[0].structure
        self.size = 1
        for s in self.structure :
            self.size *= s
        self.mu = self.init_mean.view(-1, 1).expand(-1, self.size).clone()
        self.sigma = self.init_noise.view(-1, 1).expand(-1, self.size).clone()
        self.preservation = torch.zeros((self.n, self.size), dtype=self.init_mean.dtype, device=self.init_mean.device)
        
        self.env_noise, self.env_noise_d, self.env_lamb, self.env_mean = [], [], [], []
        for env in envs :
            self.env_noise.append(env.noise)
            self.env_noise_d.append(env.noise_d)
            self.env_lamb.append(env.lamb)
            self.env_mean.append(env.mean)
        self.env_noise = torch.tensor(self.env_noise, dtype=self.init_mean.dtype).unsqueeze(-1)
        self.env_noise_d = torch.tensor(self.env_noise_d, dtype=self.init_mean.dtype).unsqueeze(-1)
        self.env_lamb = torch.tensor(self.env_lamb, dtype=self.init_mean.dtype).unsqueeze(-1)
        self.env_mean = torch.tensor(self.env_mean, dtype=self.init_mean.dtype).unsqueeze(-1)

        self.prev_reward = torch.zeros((self.n, 1))

    def setting(self, info):
        return
    
    def calculate(self, t) :
        mu = prospect(self.mu, self.loss, self.risk_plus, self.risk_minus, xi =  self.xi, prev_result = self.prev_reward)
        var = Variance(self.risk_plus, self.risk_minus, self.loss, self.mu, self.sigma, self.xi, self.prev_reward)
        sigma = torch.sqrt(var)
        return self.theta * (mu + self.rho * self.preservation + self.phi * sigma)

    def update(self, action, reward) :
        """
        def ravel_idx(array) :
            size = array[0]
            for i in range(1, len(array)) :
                size = size * self.structure[i] + array[i]
            return size
        idx = ravel_idx(action)
        """
        self.preservation = self.preservation + self.alpha * (action - self.preservation)
        K_t = torch.square(self.sigma) / (torch.square(self.sigma) + self.env_noise ** 2)
        mu_post = self.mu + action * (reward - self.mu) * K_t
        sigma_post = (1 - action)* self.sigma + action * torch.sqrt((1 - K_t) * torch.square(self.sigma))

        lamb = self.env_lamb
        noise_d = self.env_noise_d

        self.mu = lamb * mu_post + (1 - lamb) * self.env_mean
        self.sigma = torch.sqrt((lamb ** 2) * torch.square(sigma_post) + noise_d ** 2)
        self.prev_reward = reward

if __name__ == "__main__":
    with open("data/envs/restless_envs", 'rb') as f:
        restless_envs = pickle.load(f)
    #self.loss, self.risk_plus, self.risk_minus, self.init_mean, self.init_noise, self.theta, self.alpha, self.rho, self.phi, self.xi
    param = torch.tensor([5.0, 0.6, 0.6, 0, 10, 5, 0.5, 10, 10, 0])
    al, rl, r, s, p = simulate(param, restless_envs[1][0], Kalman_SM)

    print(al)
    """
    plt.figure(figsize = (12, 5))
    plt.plot(range(300), rl.detach().numpy(), color='blue',linestyle='-')
    plt.savefig('test.png')"""



"""
class GPR_SM :
    def __init__(self, params, env) :
        self.loss, self.risk, self.mean, self.noise, self.theta1, self.theta2, self.kappa, self.alpha, self.rho, self.beta1, self.beta2, self.phi2, self.phi3 = params
        self.structure = env.structure
        size = np.size(np.array(self.structure))

        gpr = GPR(kernel = RBF(length_scale=env.lamb, alpha = self.noise))
        
        self.mu = None
        self.sigma = None

        self.X, self.y = [], []

        self.preservation = np.zeros(size)
        self.actions = np.zeros(size)

        self.env = env

        self.X_every = np.stack(np.indices(self.sturcutre), axis=-1).reshape(-1, len(self.structure))  
    
    def setting(self, info):
        self.X.append(info[0])
        self.y.append(info[1])

        gpr = self.gpr.fit(self.X, self.y)
        self.mu, self.sigma = gpr.predict(self.X_every)

    def calculate(self, t) :
        theta = self.kappa * self.theta1 + (1 - self.kappa) * ((t/10) ** self.theta2)
        phi1 = np.where(self.mu >= 0, self.beta1, self.beta2)

        used_bandits = np.zeros_like(self.mu)
        for i in range(len(self.mu)) : 
            for j in range(len(self.mu)) :
                if i == j : 
                    continue
                if self.actions[i] > self.actions[j]:
                    used_bandits[i] += 1

        return theta * (self.mu + self.rho * self.preservation + phi1 * self.sigma + self.phi2 * used_bandits + self.phi3 * self.actions)

    def update(self, action, reward) :
        self.X.append(action)
        self.y.append(reward)

        idx = np.ravel_multi_index(action, self.structure)

        self.preservation[idx] = self.preservation[idx] + self.alpha * (1 - self.preservation[idx])
        self.actions = self.actions + 1
        self.actions[idx] = 0
        
        gpr = self.gpr.fit(self.X, self.y)
        self.mu, self.sigma = gpr.predict(self.X_every)
"""