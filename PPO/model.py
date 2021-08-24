'''
Neural networks model
pi: actor  ** continue action space, by Gaussian Distribution to sample **
v:  critic  ** output an estimate value V(s) **
else initialize_weights
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

"""
这里的Actor与Critic没有共享网络权重
"""


class ActorGaussian(nn.Module):
    """
    **mu** and **log_sigma**
    output:
            pi: a ~ N(mu, sigma)
            logp_a: log(probability(a | N(mu,sigma)))
    """
    def __init__(self, obs_dim, act_dim, hidden1=64, hidden2=64):
        super(ActorGaussian, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(obs_dim, act_dim)
        self.fc4 = nn.Linear(hidden1, hidden2)
        self.fc_mu = nn.Linear(hidden2, act_dim)
        self.fc_sigma = nn.Linear(hidden2, act_dim)

    def forward(self, obs, act=None):
        x = torch.tanh(self.fc1(obs))                             # obs: size*obs_dim
        pi_mu = torch.tanh(self.fc_mu(torch.tanh(self.fc2(x))))   # a_range=[-1,1]

        y = torch.tanh(self.fc3(obs))
        #y = torch.tanh(self.fc4(y))
        #log_sigma = self.fc_sigma(y)
        pi_sigma = torch.exp(y)    # important
        pi = Normal(pi_mu, pi_sigma)    # N(mu, sigma) function
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)#.sum(1)    # .sum(0:disappear row  1:disappear list )axis=-1
        return pi, logp_a


class Critic(nn.Module):
    """
    Value function, V(s)
    """
    def __init__(self, obs_dim, hidden1=64, hidden2=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.v = nn.Linear(hidden2, 1)

    def forward(self, obs):
        v = self.v(torch.tanh(self.fc2(torch.tanh(self.fc1(obs)))))
        return torch.squeeze(v, -1)        # important   [size, 1] => [size,]


class AC(nn.Module):
    """
    actor + critic
    """
    def __init__(self, obs_dim, act_dim, hidden1=64, hidden2=64):
        super().__init__()
        self.pi = ActorGaussian(obs_dim, act_dim, hidden1, hidden2)
        self.v = Critic(obs_dim, hidden1, hidden2)

    def step(self, obs):
        with torch.no_grad():
            pi, _ = self.pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)#.sum(1)
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]


def initialize_weights(network, initialization_type, scale=1.0):
    for p in network.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                torch.nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


# **************************************************************************

from torch.distributions import MultivariateNormal


class MultiNormal(nn.Module):
    def __init__(self, state_dim, action_dim, orthmatrix):
        super(MultiNormal, self).__init__()
        self.in_dim = state_dim
        self.out_dim = action_dim
        self.orthmatrix = orthmatrix
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.diag = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)
        #nn.init.constant_(self.diag.weight, 0.5)
        #nn.init.orthogonal_(self.orth_matrix.weight, 1)
        self.vfc1 = nn.Linear(state_dim, 64)
        self.vfc2 = nn.Linear(64, 64)

    def _pi_forward(self, state, device=torch.device('cpu')):
        x = torch.tanh(self.fc2(torch.tanh(self.fc1(state))))
        mu = self.mu(x)
        init_diag = self.diag(x)
        diag = torch.diag_embed(init_diag.pow(2) + 1.0*torch.ones(init_diag.shape[0], 3).to(device))
        init_orth = self.orthmatrix(x)
        orth_m = init_orth.reshape(init_orth.shape[0], self.out_dim, self.out_dim)
        sigma = orth_m.transpose(1, 2).bmm(diag).bmm(orth_m)
        pi = MultivariateNormal(mu, sigma)
        return pi

    def _v_forward(self, state):
        x = torch.tanh(self.vfc2(torch.tanh(self.vfc1(state))))
        value = self.value(x)
        return value

    def forward(self, state):
        with torch.no_grad():
            pi = self._pi_forward(state)
            a = pi.sample()
            log_p = pi.log_prob(a)
            v = self._v_forward(state)
        return a, log_p, v

    def _get_newlogp(self, state, action, device):
        pi = self._pi_forward(state, device)
        log_p = pi.log_prob(action)
        return log_p

    def act(self, state):
       return self.forward(state)[0]


class orth_matrix_Net(nn.Module):
    def __init__(self, in_put=64, out_put=9):
        super(orth_matrix_Net, self).__init__()
        self.in_put = in_put
        self.out_put = out_put
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        pred = self.fc3(x)
        return pred
