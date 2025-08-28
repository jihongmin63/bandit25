
import torch
from variable import horizon, Env
from function import prospect
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import mvn
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy import special

class Parabolic_Cylinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, x):
        out, _ = special.pbdv(v.item(), x.item())
        out = torch.tensor(out, device=v.device, dtype=v.dtype)
        ctx.save_for_backward(v, x)
        return out 
    
    @staticmethod
    def backward(ctx, grad_output):
        v, x = ctx.saved_tensors
        v0, x0 = v.item(), x.item()

        eps = 1e-5
        out1, _ = special.pbdv(v0+1, x0)
        out2, _ = special.pbdv(v0-1, x0)

        grad_x = 1/2 * (- out1 - v0 * out2)

        out3, _ = special.pbdv(v0+eps, x0)
        out4, _ = special.pbdv(v0, x0)
        
        grad_v = (out3 - out4) / eps

        grad_output_tensor = grad_output.reshape(v.shape)
        grad_v_tensor = torch.tensor(grad_v, device=v.device, dtype=v.dtype).reshape(v.shape)
        grad_x_tensor = torch.tensor(grad_x, device=v.device, dtype=v.dtype).reshape(v.shape)

        return grad_v_tensor * grad_output_tensor, grad_x_tensor * grad_output_tensor

def Variance(risk_plus, risk_minus, loss, mu, sigma, xi, reward) :
    def Helper(r, c = (mu - xi * reward) / sigma) :
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
        v2 = torch.sqrt(2 * torch.pi)
        v3 = torch.power(torch.e, - r*r / 4)
        v4 = torch.exp(torch.lgamma(r+1))
        v5 = Parabolic_Cylinder.apply(-r-1, -c)

        return v1 * v3 * v4 * v5 / v2
    
    return Helper(2 * risk_plus) + loss * loss * Helper(2 * risk_minus) - torch.square(Helper(risk_plus) - loss * Helper(risk_minus))

free_varaibles = {
    'Kalman_SM' : {'loss', 'risk_plus', 'risk_minus', 'init_mean', 'init_noise', 'theta', 'alpha', 'rho', 'phi', 'xi'}
}

class Kalman_SM :
    def __init__(self, params, env) :
        self.loss, self.risk_plus, self.risk_minus, self.init_mean, self.init_noise, self.theta, self.alpha, self.rho, self.phi, self.xi = params
        
        self.structure = env.structure
        self.size = 1
        for s in self.structure :
            self.size *= s

        self.mu = torch.ones(self.size) * self.init_mean
        self.sigma = torch.ones(self.size) * self.init_noise

        self.preservation = torch.zeros(self.size)

        self.env = env

        self.prev_reward = 0
    
    def setting(self, info):
        return
    
    def calculate(self, t) :
        mu = prospect(self.mu, self.loss, self.risk_plus, self.risk_minus, xi =  self.xi, result = self.prev_reward)
        sigma = torch.sqrt(Variance(self.risk_plus, self.risk_minus, self.loss, self.mu, self.sigma, self.xi, self.prev_reward))

        return self.theta * (mu + self.rho * self.preservation + self.phi * sigma)

    def update(self, action, reward) :
        def ravel_idx(array) :
            size = array[0]
            for i in range(1, len(array)) :
                size = size * self.structure[i] + array[i]
            return size
        idx = ravel_idx(action)

        self.preservation[idx] = self.preservation[idx] + self.alpha * (1 - self.preservation[idx])

        K_t = (torch.square(self.mu[idx])) / (torch.square(self.mu[idx]) + self.env.noise ** 2)
        
        mu_post = self.mu
        mu_post[idx] += (reward - self.mu[idx]) * K_t
        # prospect theory is utilized in decision rule
        # mu_post[idx] += (prospect(reward, self.loss, self.risk_plus, self.risk_minus) - self.mu[idx]) * K_t

        sigma_post = self.sigma
        sigma_post[action] = torch.sqrt((1 - K_t) * torch.square(self.sigma[idx]))

        self.mu = self.env.lamb * mu_post + (1 - self.env.lamb) * self.env.mean
        self.sigma = torch.sqrt(torch.square(self.env.lamb) * torch.square(sigma_post) + torch.square(self.env.noise_d))

        self.prev_reward = reward

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