import numpy as np
import torch


class NormalNoise:
    def __init__(self, action_dims, mu=0., sigma=0.1, whether_tensor=False):
        self.mu = mu
        self.sigma = sigma
        self.action_dims = action_dims
        self.whether_tensor = whether_tensor

    def __call__(self):
        mu = np.ones(self.action_dims) * self.mu
        sigma = np.ones(self.action_dims) * self.sigma
        noise = np.random.normal(mu, sigma)
        if self.whether_tensor:
            return torch.as_tensor(noise, dtype=torch.float32)
        else:
            return noise

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OrnsteinUhlenbeckNoise(object):
    def __init__(self, action_dims, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = np.zeros(action_dims)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x.clip(-0.2, 0.2)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
