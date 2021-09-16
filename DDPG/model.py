import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# For MuJoCo Environment, value input

class Actor(nn.Module):
    # Deterministic Policy
    def __init__(self, state_dims, action_dims):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dims, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dims)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))           # scale action to [-1, 1]
        return a


class Critic(nn.Module):
    # Q(s,a)
    def __init__(self, state_dims, action_dims):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dims, 400)
        self.l2 = nn.Linear(400 + action_dims, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG_net(nn.Module):
    def __init__(self, state_dims, action_dims, device, actor_lr=1e-4, critic_lr=1e-3, discount=0.99, tau=0.001):
        super(DDPG_net, self).__init__()
        self.actor = Actor(state_dims, action_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dims, action_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-2)

        self.device = device
        self.discount = discount
        self.tau = tau          # soft target update

    def select_action(self, state):
        # state input is numpy
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.actor(state.reshape(1, -1)).cpu().detach().numpy().flatten()