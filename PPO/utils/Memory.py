import torch
from torch.utils.data import Dataset
from collections import namedtuple


class Memory(object):
    """
    for on-policy algorithms
    collect {s, a, r, done, and so on} under now_pi
    to update now_pi to new_pi, and new_pi ..., until KLD(now_pi||new_pi) is too large
    """

    def __init__(self):
        self.transition = namedtuple('Transition', ('state', 'value', 'action',
                                       'logprob', 'notdone', 'reward'))
        self.memory = []       # list

    def push(self, *args):
        self.memory.append(self.transition(*args))   # namedtuple_i into list[i]

    def sample(self):
        # many namedtuple list into a large list namedtuple
        buffer = self.transition(*zip(*self.memory))
        # tuple to tensor
        states = torch.as_tensor(buffer.state, dtype=torch.float32)        # .shape:([batch_size, state_dim])
        values = torch.as_tensor(buffer.value, dtype=torch.float32)        # .shape:([batch_size])
        actions = torch.as_tensor(buffer.action, dtype=torch.float32)      # .shape:([batch_size, action_dim])
        logp_olds = torch.as_tensor(buffer.logprob, dtype=torch.float32)   # .shape: ([batch_size])
        notdones = torch.as_tensor(buffer.notdone)                         # .shape:([batch_size])
        rewards = torch.as_tensor(buffer.reward, dtype=torch.float32)      # .shape:([batch_size])
        return states, values, actions, logp_olds, notdones, rewards

    def __len__(self):
        return len(self.memory)


class DataSet(Dataset):
    """
    for training to sample iteration mini-batch
    """
    def __init__(self, s, v, a, logp_old, ret, adv):
        self.s = s
        self.v = v
        self.a = a
        self.logp_old = logp_old
        self.ret = ret
        self.adv = (adv - adv.mean()) / (adv.std()+1e-8)

    def __getitem__(self, index):
        return self.s[index, :], self.v[index], self.a[index, :], self.logp_old[index], \
                self.ret[index], self.adv[index]

    def __len__(self):
        return len(self.adv)


def RandG(memory_len, values, notdones, rewards, gamma, lamda):
    """
    compute total reward (Return) and generalization advantage estimate (GAE)
    """
    returns = torch.zeros(memory_len)    # r_to_g
    deltas = torch.zeros(memory_len)     # rt + gamma * vt+1  -vt
    GAEs = torch.zeros(memory_len)       # GAE

    # for iterating to compute
    prev_return = 0
    last_value = 0
    prev_advantage = 0

    for i in reversed(range(memory_len)):
        returns[i] = rewards[i] + gamma * prev_return * notdones[i]
        deltas[i] = rewards[i] + gamma * last_value * notdones[i] - values[i]
        # (generalization advantage estimate)
        GAEs[i] = deltas[i] + gamma * lamda * prev_advantage * notdones[i]
        prev_return = returns[i]
        last_value = values[i]
        prev_advantage = GAEs[i]

    return returns, GAEs