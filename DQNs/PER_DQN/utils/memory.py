import numpy as np

from utils.sumtree import SumTree

class Memory(object):
    '''
    Memory --call--> SumTree --call--> Replay_Buffer
    '''
    epsilon = 0.01   # small amount to avoid zero priority
    alpha = 0.6   # (p^alpha / sum(pi^alpha)),[0~1], convert the importance of TD error to priority
    beta = 0.4   # coefficient of Importance-Sampling, from initial value increasing to 1.0
    beta_increment_per_sampling = 0.000001
    abs_error_upper = 1.        # clipped abs error, also be max_p

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    # --- self.tree.data.store_frame(frame)
    # --- self.tree.data.encode_recent_observation(idx)
    # --- self.trss.data.store_effect(action, reward, done)

    def store(self):
        # when interact with Env
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_error_upper
        self.tree.add(max_p)
    # every time call it, do: self.tree.data_pointer += 1

    def sample(self, batch_size):
        sample_times = 1
        mini_idx = []
        tree_idx = np.zeros((batch_size), dtype=np.int32)
        ISweights = np.empty((batch_size, 1))
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling / sample_times])
        # priority_segment
        pri_seg = (self.tree.total_p() - self.tree.tree[-1]) / batch_size

        min_prob = np.min(self.tree.tree[(self.tree.capacity-1):(self.tree.capacity+self.tree.data.num_in_buffer-1)]) / self.tree.total_p()

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i+1)
            v = np.random.uniform(a, b)
            idx, p, data_idx = self.tree.get_leaf(v)
            tree_idx[i] = idx
            prob = p / self.tree.total_p()
            ISweights[i, 0] = np.power(prob / min_prob, -self.beta)
            mini_idx.append(data_idx)

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.tree.data.sample(batch_size, mini_idx)
        return tree_idx, ISweights, obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def batch_update(self, tree_idx, abs_errors):    # *** abs_errors
        # batch data to update priority of node, which choosed to update Q net.
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_error_upper)
        p_alpha = np.power(clipped_errors, self.alpha)
        for t_i, p in zip(tree_idx, p_alpha):
            self.tree.update(t_i, p)




