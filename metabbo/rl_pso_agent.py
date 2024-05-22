from torch import nn
from torch.distributions import Normal
from metabbo.basic_agent import Basic_Agent
from metabbo.networks import MLP
import collections
import torch
import random
import numpy as np
import pickle
import os


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


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


def save_class(dir, file_name, saving_class):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(saving_class, f, -1)


class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()

        net_config = [{'in': config.feature_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 8, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]

        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)

        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, require_entropy=False, require_musigma=False):
        mu = self.__mu_net(x_in)
        mu = (torch.tanh(mu) + 1.) / 2.
        sigma = self.__sigma_net(x_in)
        sigma = (torch.tanh(sigma) + 1.) / 2.
        sigma = torch.clamp(sigma, min=self.__min_sigma, max=self.__max_sigma)

        policy = Normal(mu, sigma)
        action = policy.sample()

        filter = torch.abs(action - 0.5) >= 0.5
        action = torch.where(filter, (action + 3 * sigma.detach() - mu.detach()) * (1. / 6 * sigma.detach()), action)
        log_prob = policy.log_prob(action)

        if require_entropy:
            entropy = policy.entropy()

            out = (action, log_prob, entropy)
        else:
            if require_musigma:
                out = (action, log_prob, mu, sigma)
            else:
                out = (action, log_prob)

        return out


class RL_PSO_Agent(Basic_Agent):
    def __init__(self, config, use_feature_extractor = False):
        super().__init__(config)
        # add specified config
        if use_feature_extractor:
            config.feature_dim = config.hidden_dim
        else:
            if config.use_ela:
                config.feature_dim = 32
            else:
                config.feature_dim = 2*config.dim # if use auto feature extractor feature dim is 16, else 2*dim
        config.action_dim = 1
        config.action_shape = (1,)
        config.max_sigma = 0.7
        config.min_sigma = 0.01
        config.lr = 1e-5
        self.__config = config
        self.__device = config.device
        self.__nets = PolicyNetwork(config).to(self.__device)

        # optimizer
        self.__optimizer = torch.optim.Adam([{'params': self.__nets.parameters(), 'lr': config.lr}])
        self.__learning_time = 0



    def train_episode(self, env):

        # input action_dim should be : bs, ps
        # action in (0,1) the ratio to learn from pbest & gbest
        state = env.reset()
        state = torch.FloatTensor(state).to(self.__device)
        
        exceed_max_ls = False
        R = 0
        while True:
            action, log_prob = self.__nets(state)
            action = action.reshape(self.__config.action_shape)
            action = action.cpu().numpy()
            
            state, reward, is_done = env.step(action)
            R += reward
            state = torch.FloatTensor(state).to(self.__device)
            
            policy_gradient = -log_prob*reward
            loss = policy_gradient.mean()

            self.__optimizer.zero_grad()
            loss.mean().backward()
            self.__optimizer.step()
            self.__learning_time += 1

            # if self.__learning_time >= self.__config.max_learning_step:
            #     exceed_max_ls = True
            #     break

            if is_done:
                break
        return exceed_max_ls, {'normalizer': env.optimizer.cost[0],
                               'gbest': env.optimizer.cost[-1],
                               'return': R,
                               'learn_steps': self.__learning_time}
    
    def rollout_episode(self, env):
        with torch.no_grad():
            is_done = False
            state = env.reset()
            R=0
            while not is_done:
                state = torch.FloatTensor(state)
                action, _ = self.__nets(state)
                state, reward, is_done = env.step(action.cpu().numpy())
                R+=reward
            return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}
