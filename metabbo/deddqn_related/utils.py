import collections
import torch
import random
import numpy as np



class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        
        obs_batch = torch.FloatTensor(np.array(obs_batch))
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
        done_batch = torch.FloatTensor(done_batch)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)