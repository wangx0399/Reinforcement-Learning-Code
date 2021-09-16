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
    # Q1(s,a)  Q2(s,a)
    def __init__(self, state_dims, action_dims):
        super(Critic, self).__init__()
        self.l11 = nn.Linear(state_dims + action_dims, 400)
        self.l12 = nn.Linear(400 + action_dims, 300)
        self.l13 = nn.Linear(300, 1)

        self.l21 = nn.Linear(state_dims + action_dims, 400)
        self.l22 = nn.Linear(400 + action_dims, 300)
        self.l23 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l11(torch.cat([state, action], 1)))
        q1 = F.relu(self.l12(torch.cat([q1, action], 1)))
        q1 = self.l13(q1)

        q2 = F.relu(self.l21(torch.cat([state, action], 1)))
        q2 = F.relu(self.l22(torch.cat([q2, action], 1)))
        q2 = self.l23(q2)
        return q1, q2

    def q_update_pi(self, state, action):
        q1 = F.relu(self.l11(torch.cat([state, action], 1)))
        q1 = F.relu(self.l12(torch.cat([q1, action], 1)))
        q1 = self.l13(q1)
        return q1


class TD3_net(nn.Module):
    def __init__(self, state_dims, action_dims, device, actor_lr=1e-3, critic_lr=1e-3, discount=0.99, tau=0.005):
        super(TD3_net, self).__init__()
        self.actor = Actor(state_dims, action_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dims, action_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)


        self.device = device
        self.discount = discount
        self.tau = tau          # soft target update

    def select_action(self, state):
        # state input is numpy
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.actor(state.reshape(1, -1)).cpu().detach().numpy().flatten()