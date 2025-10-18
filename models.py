
import torch
from variable import horizon, Env
from function import prospect, simulate
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

class Parabolic_Cylinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        v_np = v.detach().cpu().numpy().astype(np.float64)
        x_np = x.detach().cpu().numpy().astype(np.float64)
        out_np, _ = special.pbdv(v_np, x_np)
        out = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)
        ctx.save_for_backward(v, x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, x = ctx.saved_tensors
        v0 = v.detach().cpu().numpy().astype(np.float64)
        x0 = x.detach().cpu().numpy().astype(np.float64)

        eps = 1e-8

        _, grad_x_np = special.pbdv(v0, x0)

        out3, _ = special.pbdv(v0 + eps, x0)
        out4, _ = special.pbdv(v0 - eps, x0)
        grad_v_np = (out3 - out4) / (2 * eps)

        grad_x = torch.from_numpy(grad_x_np).to(device=x.device, dtype=x.dtype)
        grad_v = torch.from_numpy(grad_v_np).to(device=v.device, dtype=v.dtype)

        return grad_v * grad_output, grad_x * grad_output
    
def Variance(risk_plus, risk_minus, loss, mu, sigma, xi, reward) :
    c = (mu - xi * reward) / sigma
    def Helper(r, c, log = 0) :
        """
        Helper function calculating distorted sigma
        Args :
            r (float) : risk averse parameter; alpha or beta
            c (array) : mu / sigma for bandits
            sigma (array) : sigma of bandits

        Returns:
            (array) 
        """
        v1 = torch.pow(sigma, r)
        v2 = torch.sqrt(torch.tensor(2 * torch.pi, device=xi.device, dtype=r.dtype))
        v3 = torch.exp(- (c*c) / 4)
        v4 = torch.exp(torch.lgamma((r+1)))
        v5 = Parabolic_Cylinder.apply(-r-1, -c)
        if (log) :
            print(" r, c : ", r, c)
            print(" v1 : ", v1)
            print(" v2 : ", v2)
            print(" v3 : ", v3)
            print(" v4 : ", v4)
            print(" v5 : ", v5)
        return v1 * v3 * v4 * v5  / v2
    var = Helper(2 * risk_plus, c) + loss * loss * Helper(2 * risk_minus, -c) - torch.square(Helper(risk_plus, c) - loss * Helper(risk_minus, -c))
    
    with torch.no_grad() :
        var = torch.clamp(var, min=0)
    
    return var


free_varaibles = {
    'Kalman_SM' : ['loss', 'risk_plus', 'risk_minus', 'init_mean', 'init_noise', 'theta', 'alpha', 'rho', 'phi', 'xi']
}

class Kalman_SM :
    def __init__(self, params, env) :
        self.loss, self.risk_plus, self.risk_minus, self.init_mean, self.init_noise, self.theta, self.alpha, self.rho, self.phi, self.xi = params
        self.structure = env.structure
        self.size = 1
        for s in self.structure :
            self.size *= s

        self.mu = self.init_mean.expand(self.size).clone()
        self.sigma = self.init_noise.expand(self.size).clone()
        self.preservation = torch.zeros(self.size, dtype=self.init_mean.dtype, device=self.init_mean.device)


        self.env = env
        self.prev_reward = 0
    
    def setting(self, info):
        return
    
    def calculate(self, t, debug = 0) :
        mu = prospect(self.mu, self.loss, self.risk_plus, self.risk_minus, xi =  self.xi, prev_result = self.prev_reward)
        #var = Variance(self.risk_plus, self.risk_minus, self.loss, self.mu, self.sigma, self.xi, self.prev_reward)
        #sigma = torch.sqrt(var)
        sigma = self.sigma
        #if debug :
            #print(mu, var, sigma)
        return self.theta * (mu + self.rho * self.preservation + self.phi * sigma)

    def update(self, action, reward, debug = 0) :
        """
        def ravel_idx(array) :
            size = array[0]
            for i in range(1, len(array)) :
                size = size * self.structure[i] + array[i]
            return size
        idx = ravel_idx(action)
        """
        if debug :
            print(" - prev pre ", self.preservation)
            print(" - prev mu ", self.mu)
            print(" - prev sigma ", self.sigma)
        self.preservation = self.preservation + self.alpha * (action - self.preservation)
        K_t = torch.square(self.sigma) / (torch.square(self.sigma) + self.env.noise ** 2)

        mu_post = self.mu + action * (reward - self.mu) * K_t
        sigma_post = (1 - action) * self.sigma + action * torch.sqrt((1 - K_t) * torch.square(self.sigma))

        lamb = self.env.lamb
        noise_d = self.env.noise_d

        self.mu = lamb * mu_post + (1 - lamb) * self.env.mean
        self.sigma = torch.sqrt((lamb ** 2) * torch.square(sigma_post) + noise_d ** 2)
        self.prev_reward = reward

        if debug :
            print(" - int mu ", mu_post)
            print(" - int sigma ", sigma_post)
            print(" - cur pre ", self.preservation)
            print(" - cur mu ", self.mu)
            print(" - cur sigma ", self.sigma)

        

if __name__ == "__main__":
    with open("envs/restless_envs", 'rb') as f:
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