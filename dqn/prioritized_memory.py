# From: https://github.com/rlcode/per

import random
import numpy as np
from SumTree import SumTree
from collections import namedtuple
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, a=None, beta=None):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        if a:
            self.a = a
        if beta:
            self.beta = beta

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        p = self._get_priority(error)
        self.tree.add(p, e)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        # Gather the batch data
        try:
            states = torch.from_numpy(np.vstack([e.state for e in batch if e ])).float().to(device)
        except (AttributeError,ValueError) as e:
            import pdb; pdb.set_trace()
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e ])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e ])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e ])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e ]).astype(np.uint8)).float().to(device)
        
        # Redo idxs and is_weight to match length of states, etc
        idxs = [ idxs[i] for i,e in enumerate(batch) if e ]
        is_weight = np.array([ is_weight[i] for i,e in enumerate(is_weight.tolist()) if e ])
        
        # I don't exactly understand why an experience can be 0
        

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
