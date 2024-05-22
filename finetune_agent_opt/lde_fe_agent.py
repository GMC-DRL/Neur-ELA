import copy
import torch.nn as nn
from metabbo.basic_agent import Basic_Agent
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


class PolicyNet(nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        self.__lstm = nn.LSTM(input_size=config.node_dim,
                              hidden_size=config.CELL_SIZE,
                              num_layers=config.LAYERS_NUM).to(config.device)
        self.__mu = nn.Linear(config.CELL_SIZE, config.output_dim_actor).to(config.device)
        self.__sigma = nn.Linear(config.CELL_SIZE, config.output_dim_actor).to(config.device)
        self.__distribution = torch.distributions.Normal
        self.__config = config

    def forward(self, x, h, c):
        cell_out, (h_, c_) = self.__lstm(x, (h, c))
        mu = self.__mu(cell_out)
        sigma = torch.sigmoid(self.__sigma(cell_out))
        return mu, sigma, h_, c_

    def sampler(self, inputs, ht, ct):
        mu, sigma, ht_, ct_ = self.forward(inputs, ht, ct)
        normal = self.__distribution(mu, sigma)
        sample_w = torch.clip(normal.sample(), 0, 1).reshape(self.__config.action_shape)
        return sample_w, ht_, ct_


class LDE_FE_Agent(Basic_Agent):
    def __init__(self, config, fe=None):
        super().__init__(config)
        self.__config = config
        self.__BATCH_SIZE = 1
        self.__config.NP = 50
        self.__config.TRAJECTORY_NUM = 20
        self.__config.TRAJECTORY_LENGTH = 50
        self.__config.CELL_SIZE = 50
        self.__config.BINS = 5
        self.__config.LAYERS_NUM = 1
        self.__config.lr_model = 0.005
        self.__config.lr_decay = 1
        self.__config.gamma = 0.99
        self.__config.output_dim_actor = self.__config.NP * 2
        self.__config.action_shape = (1, self.__BATCH_SIZE, self.__config.NP * 2,)
        self.fe = fe
        if self.fe is not None:
            self.__config.node_dim = 16
        else:
            self.__config.node_dim = self.__config.NP + 2 * self.__config.BINS
        self.__feature_shape = (self.__BATCH_SIZE, self.__config.node_dim,)

        self.__net = PolicyNet(self.__config)

        self.__optimizer = torch.optim.Adam(
                                        [{'params': self.fe.parameters(), 'lr': self.__config.lr_model}] +
                                        [{'params':self.__net.parameters(),'lr':self.__config.lr_model}])
        self.__learn_steps = 0


    def update_setting(self, config):
        self.__config.max_learning_step = config.max_learning_step
        #self.__config.agent_save_dir = config.agent_save_dir
        self.__learn_steps = 0
        # save_class(self.__config.agent_save_dir, 'checkpoint0', self)
        # self.__cur_checkpoint = 1

    def __discounted_norm_rewards(self, r):
        for ep in range(self.__config.TRAJECTORY_NUM * self.__BATCH_SIZE):
            length = r.shape[0] // self.__config.TRAJECTORY_NUM
            single_rs = r[ep * length: ep * length + length]
            discounted_rs = np.zeros_like(single_rs)
            running_add = 0.
            for t in reversed(range(0, length)):
                running_add = running_add * self.__config.gamma + single_rs[t]
                discounted_rs[t] = running_add
            if ep == 0:
                all_disc_norm_rs = discounted_rs
            else:
                all_disc_norm_rs = np.hstack((all_disc_norm_rs, discounted_rs))
        return all_disc_norm_rs

    def train_episode(self, env):
        if env.optimizer.fe is not None:
            env.optimizer.fe.set_on_train()
        self.__optimizer.zero_grad()
        inputs_batch = []
        action_batch = []
        hs_batch = []
        cs_batch = []
        rewards_batch = []
        R = 0
        for l in range(self.__config.TRAJECTORY_NUM):
            input_net = env.reset()
            h0 = torch.zeros(self.__config.LAYERS_NUM, self.__BATCH_SIZE, self.__config.CELL_SIZE).to(self.__config.device)
            c0 = torch.zeros(self.__config.LAYERS_NUM, self.__BATCH_SIZE, self.__config.CELL_SIZE).to(self.__config.device)
            is_done = False
            for t in range(self.__config.TRAJECTORY_LENGTH):
                
                input_net = input_net.reshape(self.__feature_shape)
                # [bs, NP+BINS*2]
                action, h_, c_ = self.__net.sampler(torch.FloatTensor(input_net[None, :]).to(self.__config.device), h0, c0)  # parameter controller
                action = action.reshape(1, self.__BATCH_SIZE, -1)
                action = np.squeeze(action.cpu().numpy(), axis=0)
                inputs_batch.append(input_net)
                
                action_batch.append(action)
                next_input, reward, is_done = env.step(action)
                
                hs_batch.append(torch.squeeze(h0, axis=0))
                cs_batch.append(torch.squeeze(c0, axis=0))
                rewards_batch.append(reward.reshape(self.__BATCH_SIZE))
                R += np.mean(reward)
                h0 = h_
                c0 = c_
                input_net = next_input
                if is_done:
                    break
        inputs = [torch.stack(inputs_batch, axis=0).permute(1, 0, 2).reshape(-1, self.__config.node_dim)]
        actions = [np.stack(action_batch, axis=0).transpose((1, 0, 2)).reshape(-1, self.__config.output_dim_actor)]
        hs = [torch.stack(hs_batch, axis=0).permute(1, 0, 2).reshape(-1, self.__config.CELL_SIZE)]
        cs = [torch.stack(cs_batch, axis=0).permute(1, 0, 2).reshape(-1, self.__config.CELL_SIZE)]
        rewards = [np.stack(rewards_batch, axis=0).transpose((1, 0)).flatten()]

        # update network parameters
        all_eps_mean, all_eps_std, all_eps_h, all_eps_c = self.__net.forward(torch.vstack(inputs)[None, :],
                                                                    torch.vstack(hs)[None, :],
                                                                    torch.vstack(cs)[None, :])
        actions = torch.FloatTensor(np.vstack(actions)).to(self.__config.device)
        all_eps_mean = torch.squeeze(all_eps_mean, 0).to(self.__config.device)
        all_eps_std = torch.squeeze(all_eps_std, 0).to(self.__config.device)
        normal_dis = torch.distributions.Normal(all_eps_mean, all_eps_std)
        log_prob = torch.sum(normal_dis.log_prob(actions + 1e-8), 1).to(self.__config.device)
        all_eps_dis_reward = self.__discounted_norm_rewards(np.hstack(rewards))
        loss = - torch.mean(log_prob * torch.FloatTensor(all_eps_dis_reward).to(self.__config.device))
        loss.backward()
        self.__optimizer.step()
        self.__learn_steps += 1

        # if self.__learn_steps >= (self.__config.save_interval * self.__cur_checkpoint):
        #     save_class(self.__config.agent_save_dir,'checkpoint'+str(self.__cur_checkpoint),self)
        #     self.__cur_checkpoint+=1

        return self.__learn_steps >= self.__config.max_learning_step, {'normalizer': env.optimizer.cost[0],
                                                                       'gbest': env.optimizer.cost[-1],
                                                                       'return': R,
                                                                       'learn_steps': self.__learn_steps}

    def rollout_episode(self, env):
        if env.optimizer.fe is not None:
            env.optimizer.fe.set_off_train()
        with torch.no_grad():
            is_done = False
            input_net = env.reset()
            h0 = torch.zeros(self.__config.LAYERS_NUM, self.__BATCH_SIZE, self.__config.CELL_SIZE).to(self.__config.device)
            c0 = torch.zeros(self.__config.LAYERS_NUM, self.__BATCH_SIZE, self.__config.CELL_SIZE).to(self.__config.device)
            R=0
            while not is_done:
                # [bs, NP+BINS*2]
                action, h_, c_ = self.__net.sampler(torch.FloatTensor(input_net[None, :]).to(self.__config.device), h0, c0)  # parameter controller
                action = action.reshape(1, self.__BATCH_SIZE, -1)
                action = np.squeeze(action.cpu().numpy(), axis=0)
                next_input, reward, is_done = env.step(action)
                R+=np.mean(reward)
                h0 = h_
                c0 = c_
                input_net = copy.deepcopy(next_input)
            return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes,'return':R}
