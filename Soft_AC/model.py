import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.distributions.normal import Normal
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(Actor, self).__init__()
        self.com = nn.Linear(state_dims, 256)
        self.mu1 = nn.Linear(256, 128)
        self.mu2 = nn.Linear(128, action_dims)
        self.sigma1 = nn.Linear(256, 128)
        self.sigma2 = nn.Linear(128, action_dims)

    def forward(self, state):
        com = F.relu(self.com(state))
        mu = torch.tanh(self.mu2(F.relu(self.mu1(com))))     # F.tanh()  for MuJoCo action range
        log_sigma = self.sigma2(F.relu(self.sigma1(com)))
        sigma = torch.exp(log_sigma)
        return mu, sigma                                 # mean and standard deviation


class Critic(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(Critic, self).__init__()
        self.l11 = nn.Linear(state_dims, 256)
        self.l12 = nn.Linear(256 + action_dims, 128)
        self.l13 = nn.Linear(128, 1)

        self.l21 = nn.Linear(state_dims, 256)
        self.l22 = nn.Linear(256 + action_dims, 128)
        self.l23 = nn.Linear(128, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l11(state))
        q1 = F.relu(self.l12(torch.cat([q1, action], 1)))
        q1 = self.l13(q1)

        q2 = F.relu(self.l21(state))
        q2 = F.relu(self.l22(torch.cat([q2, action], 1)))
        q2 = self.l23(q2)
        return q1, q2


class SAC_net(nn.Module):
    def __init__(self, state_dims, action_dims, max_action, device,
                 actor_lr=3e-4, critic_lr=3e-4, discount=0.99, tau=0.005):
        super(SAC_net, self).__init__()
        self.actor = Actor(state_dims, action_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dims, action_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.zeros(1).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optim = torch.optim.SGD([self.log_alpha], lr=0.01, momentum=0.9)

        self.max_action = max_action
        self.device = device
        self.discount = discount
        self.tau = tau

    def interact_action(self, state):
        # when collect data, input state is ndarry
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mu, sigma = self.actor(state.to(self.device))
            actor = Normal(mu, sigma)
            action = torch.tanh(actor.rsample()) * self.max_action
        return action.cpu().numpy()

    def get_log_p(self, states):
        # when sample minibatch to train, here state is tensor
        mus, sigmas = self.actor(states)
        actor = Normal(mus, sigmas)
        actions = actor.rsample()
        log_p_as = actor.log_prob(actions)
        log_p_as = (log_p_as - 2*(np.log(2.) - actions - F.softplus(-2*actions))).sum(1)
        actions = torch.tanh(actions) * self.max_action
        return actions, log_p_as.unsqueeze(-1)
