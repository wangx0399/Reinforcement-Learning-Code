import numpy as np
import torch


class ReplayBuffer(object):
	# a first in first out Replay Buffer
	def __init__(self, state_dims, action_dims, max_size=int(1e6)):
		self.max_size = max_size
		self.pointer = 0
		self.can_sample_size = 0
		#order list   {s, a, r, s', done}
		self.state = np.zeros((max_size, state_dims))
		self.action = np.zeros((max_size, action_dims))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dims))
		self.not_done = np.zeros((max_size, 1)).astype(np.int)

	def add(self, state, action, reward, next_state, mask):
		self.state[self.pointer] = state
		self.action[self.pointer] = action
		self.reward[self.pointer] = reward
		self.next_state[self.pointer] = next_state
		self.not_done[self.pointer] = 1. - mask

		self.pointer = (self.pointer + 1) % self.max_size
		self.can_sample_size = min(self.can_sample_size+1, self.max_size)

	def random_sample(self, batch_size):
		index = np.random.randint(0, self.can_sample_size, batch_size)
		return (
			torch.as_tensor(self.state[index], dtype=torch.float32),
			torch.as_tensor(self.action[index], dtype=torch.float32),
			torch.as_tensor(self.reward[index], dtype=torch.float32),
			torch.as_tensor(self.next_state[index], dtype=torch.float32),
			torch.as_tensor(self.not_done[index], dtype=torch.float32)
		)
