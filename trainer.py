import pickle
import torch
import numpy as np

from metabbo.basic_environment import PBO_Env

from metabbo.lde_agent import LDE_Agent
from metabbo.rl_pso_agent import RL_PSO_Agent
from metabbo.rlepso_agent import RLEPSO_Agent
from metabbo.lde_optimizer import LDE_Optimizer
from metabbo.rlepso_optimizer import RLEPSO_Optimizer
from metabbo.rl_pso_optimizer import RL_PSO_Optimizer
from metabbo.gleet_agent import GLEET_Agent
from metabbo.gleet_optimizer import GLEET_Optimizer
from metabbo.rl_das_agent import RL_DAS_Agent
from metabbo.rl_das_optimizer import RL_DAS_Optimizer
from metabbo.deddqn_agent import DE_DDQN_Agent
from metabbo.deddqn_optimizer import DE_DDQN_Optimizer

import pickle
import math

from tqdm import tqdm
from eval.fitness import calculate_per_task_perf, calculate_aggregate_performance
from eval.cost_baseline import get_train_cost_baseline
import copy
import os




class Trainer(object):
    def __init__(self, config, train_set, test_set, seed, fe=None):
        self.config = config
        torch.manual_seed(seed)
        np.random.seed(seed)
        if fe is None:
            self.agent = eval(config.train_agent)(config, False)
            self.optimizer = eval(config.train_optimizer)(config, fe)
        else:
            self.agent = eval(config.train_agent)(config, True)
            self.optimizer = eval(config.train_optimizer)(config, fe)
        self.train_set = train_set
        self.test_set = test_set
        self.fe = fe
        

    def train(self, pick_best=False, trival=False, save_checkpoint = False):
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
        

        while not exceed_max_lt:
            cost_one_episode = {}
            with tqdm(range(self.train_set.N), desc=f'Training {self.config.train_agent} Epoch {epoch}') as pbar:
                for problem in self.train_set:
                    env = PBO_Env(problem, self.optimizer)
                    _, info = self.agent.train_episode(env)  # pbar_info -> dict
                    cost_one_episode[problem.__str__()] = [info['gbest']]
                    # pbar.set_postfix(pbar_info_train)
                    pbar.update(1)
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
            if save_checkpoint:
                with open(os.path.join(self.config.save_checkpoint_dir, f'epoch-{epoch}.pkl'), 'wb') as f:
                    pickle.dump(self, f, -1)

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
            if not self.config.use_ela:
                with open(f'records/trival_saved/{self.config.train_agent}_{self.config.train_epoch}_{self.config.dataset}.pkl', 'wb') as f:
                    pickle.dump(self, f, -1)
            else:
                with open(f'records/trival_saved/{self.config.train_agent}_{self.config.train_epoch}_ela_count{self.config.count_ela_fes}_{self.config.dataset}.pkl', 'wb') as f:
                    pickle.dump(self, f, -1)
        record = self.rollout(agent=self if not pick_best else best_agent)
        
        return record
    
    def rollout(self, agent):
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

