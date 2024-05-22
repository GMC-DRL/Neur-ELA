import pickle
import torch
from metabbo.basic_environment import PBO_Env
import numpy as np

from finetune_agent_opt.rl_das_fe_agent import RL_DAS_FE_Agent
from finetune_agent_opt.rl_das_fe_optimizer import RL_DAS_FE_Optimizer
from finetune_agent_opt.lde_fe_agent import LDE_FE_Agent
from finetune_agent_opt.lde_fe_optimizer import LDE_FE_Optimizer

import pickle
import math
from eval.fitness import *
# from ray.experimental.tqdm_ray import tqdm
from tqdm import tqdm
from eval.cost_baseline import get_train_cost_baseline
import copy
import os
# load from .pkl file



class Trainer(object):
    def __init__(self, config, train_set, test_set, seed, fe=None):
        self.config = config
        torch.manual_seed(seed)
        np.random.seed(seed)
        if fe is None:
            self.agent = eval(config.train_agent)(config, fe)
            self.optimizer = eval(config.train_optimizer)(config, fe)
        else:
            self.agent = eval(config.train_agent)(config, fe)
            self.optimizer = eval(config.train_optimizer)(config, fe)
        self.train_set = train_set
        self.test_set = test_set

        self.fe = fe
        self.cur_checkpoint = 0

    def train(self, pick_best=False, trival=False):
        print('start training')
        exceed_max_lt = False
        # start = time.time()
        epoch = 0
        if pick_best:
            best_agent = None
            best_perf = math.inf
        # # for trival
        if trival:
            cost_trival_train = []
        collect_rollout_perf = []
        # before training
        # collect_rollout_perf.append(self.rollout(agent=self))
        while not exceed_max_lt:
            cost_one_episode = {}
            with tqdm(range(self.train_set.N), desc=f'Training {self.config.train_agent} Epoch {epoch}') as pbar:
                for problem in self.train_set:
                    env = PBO_Env(problem, self.optimizer)
                    _, info = self.agent.train_episode(env)  # pbar_info -> dict
                    cost_one_episode[problem.__str__()] = [info['gbest']]
                    # pbar.set_postfix(pbar_info_train)
                    pbar.update(1)

                    # ! debug to delete
                    for v in self.agent.fe.parameters():
                        print(v.data)
                        break
                    # now = time.time()
            if pick_best:
                task_perf = calculate_per_task_perf(raw_data=cost_one_episode, fitness_mode=self.config.fitness_mode, cost_baseline=get_train_cost_baseline(self.config.dataset)[self.config.train_agent[:-len('_Agent')]])
                perf = calculate_aggregate_performance(task_performance_results=[{'raw_data': cost_one_episode, 'task_perf': task_perf}], agent_list=[self.config.train_agent[:-len('_Agent')]], in_task_agg=self.config.in_task_agg, out_task_agg=self.config.out_task_agg)['final_score']
                # for debug
                print(f'perf: {perf}\n task_perf: {task_perf}')
                if perf <= best_perf:
                    best_agent = copy.deepcopy(self)
                    best_perf = perf
                    # just for debug
                    # print(f'Update the best agent, cur_best_perf:{best_perf}, cur_best_epoch:{epoch}')
            # save checkpoint
            with open(os.path.join(self.config.save_checkpoint_dir, f'epoch-{epoch}.pkl'), 'wb') as f:
                pickle.dump(self, f, -1)
            if epoch >= self.cur_checkpoint * self.config.train_epoch / 20:
                record = self.rollout(agent=self)
                collect_rollout_perf.append(record)
                self.cur_checkpoint += 1
            # just for trival
            if trival:
                if epoch > self.config.train_epoch - 3:
                    cost_trival_train.append(cost_one_episode)
            epoch += 1
            
            if epoch > self.config.train_epoch: # an agent can learn no more than 20 min
                exceed_max_lt = True
                # # just for trival
                if trival:
                    print_result = {}
                    for k in cost_trival_train[0].keys():
                        print_result[k] = [cost_trival_train[i][k][0] for i in range(len(cost_trival_train))]
                    print(f'Agent: {self.config.train_agent}, train_result: {print_result}')
        # for trival
        if trival:
            with open(f'records/trival_saved/{self.config.train_agent}_{self.config.train_epoch}.pkl', 'wb') as f:
                pickle.dump(self, f, -1)
        
        record = self.rollout(agent=self if not pick_best else best_agent)
        collect_rollout_perf.append(record)
        # save record
        with open(os.path.join(self.config.save_checkpoint_dir, 'collect.pkl'), 'wb') as f:
            pickle.dump(collect_rollout_perf, f, -1)
        return record
    
    def rollout(self, agent):
        # todo: choose the best agent, this is really infeasible
        print('start testing')
        cost_record = {}
        # with tqdm(range(self.test_set.N)) as pbar:
        for problem in self.test_set:
            cost_record[problem.__str__()] = []
            for i in range(5): # For each problem we test 3 runs to average
                torch.manual_seed(i)
                np.random.seed(i)
                env = PBO_Env(problem, agent.optimizer)
                best_found_obj = agent.agent.rollout_episode(env)['cost'][-1]
                cost_record[problem.__str__()].append(best_found_obj)
            # pbar.update(1)
        return cost_record