
import numpy as np
from variable import horizon, Env
from function import prospect
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import mvn
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import gamma, pbdv

def pdf(x, mu, sigma) :
    return norm.pdf(x, loc=mu, scale=sigma)
    """
    eps = 1e-12
    z = (x - mu) / sigma
    p = norm.cdf(z)
    p = np.clip(p, eps, 1.0 - eps)
    return np.exp(- eta * np.power(-np.log(p), alpha)) * eta * alpha * np.power(-np.log(p), alpha - 1) / sigma
    """

def Variance(risk_plus, risk_minus, loss, mu, sigma) :
    def Helper(r, c = mu / sigma) :
        """
        Helper function calculating distorted sigma
        Args :
            r (float) : risk averse parameter; alpha or beta
            c (array) : mu / sigma for bandits
            sigma (array) : sigma of bandits

        Returns:
            (array) 
        """

        v1 = np.power(sigma, r)
        v2 = np.sqrt(2 * np.pi())
        v3 = np.power(np.e, - r*r / 4)
        v4 = gamma(z = r+1)
        v5, _ = pbdv(v = -r-1, x = -c)

        return v1 * v3 * v4 * v5 / v2
    
    return Helper(2 * risk_plus) + loss * loss * Helper(2 * risk_minus) - np.square(Helper(risk_plus) - loss * Helper(risk_minus))

class Kalman_SM :
    def __init__(self, params, env) :
        self.loss, self.risk_plus, self.risk_minus, self.mean, self.noise, self.theta, self.alpha, self.rho, self.phi = params
        
        self.structure = env.structure
        self.size = 1
        for s in self.structure :
            self.size *= s
        size = self.size

        self.mu = np.ones(size) * self.mean
        self.sigma = np.ones(size) * self.noise

        self.preservation = np.zeros(size)

        self.env = env
    
    def setting(self, info):
        return
    
    def calculate(self, t) :
        mu = prospect(self.mu, self.loss, self.risk_plus, self.risk_minus)
        sigma = np.sqrt(Variance(self.risk_plus, self.risk_minus, self.loss, self.mu, self.sigma))

        return self.theta * (mu + self.rho * self.preservation + self.phi * sigma)

    def update(self, action, reward) :
        idx = np.ravel_multi_index(action, self.structure)

        self.preservation[idx] = self.preservation[idx] + self.alpha * (1 - self.preservation[idx])

        K_t = (np.square(self.mu[idx])) / (np.square(self.mu[idx]) + self.env.noise ** 2)
        
        mu_post = self.mu
        mu_post[idx] += (reward - self.mu[idx]) * K_t
        # prospect theory is utilized in decision rule
        # mu_post[idx] += (prospect(reward, self.loss, self.risk_plus, self.risk_minus) - self.mu[idx]) * K_t

        sigma_post = self.sigma
        sigma_post[action] = np.sqrt((1 - K_t) * np.square(self.sigma[idx]))

        self.mu = self.env.lamb * mu_post + (1 - self.env.lamb) * self.env.mean
        self.sigma = np.sqrt(np.square(self.env.lamb) * np.square(sigma_post) + np.square(self.env.noise_d))

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