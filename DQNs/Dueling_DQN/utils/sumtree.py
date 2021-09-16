import math
import numpy as np
from utils.replay_buffer import Replay_Buffer

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity         # leaf number = the buffer size
        self.data_pointer = 0
        self.tree = np.zeros(2 * capacity - 1)      # the total number of nodes, and store prioritize
        self.data = Replay_Buffer(capacity, frame_history_len=4)     # store transition of leaf node

    def add_frame(self, data_frame):
        return self.data.store_frame(self.data_pointer, data_frame)

    def give_state(self):
        return self.data.encode_recent_observation(self.data_pointer)

    def add_effect(self, action, reward, done):
        return self.data.store_effect(self.data_pointer, action, reward, done)

    def add(self, p):
        # p is prioritize value, TD-error
        # data is {s, a, r, s', done}
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # Propagate the change through tree until node 0
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        # by sampled p, choose which leaf node
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1    # child_left_node_index
            cr_idx = cl_idx + 1            # child_right_node_index
            if cl_idx >= (2*self.capacity-1):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_p(self):
        return self.tree[0]     # the root node