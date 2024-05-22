import torch
import copy
from config import get_config
from trainer import Trainer
import pickle
from dataset.bbob import *
from dataset.protein_docking import *
import warnings
warnings.filterwarnings('ignore')
from eval.fitness import *
from eval.cost_baseline import get_test_cost_baseline
from eval.utils import *



config = get_config()
config.dataset = 'protein_docking'
train_set, test_set = construct_problem_set(dataset=config.dataset)


if config.dataset == 'protein_docking':
    config.dim = 12
    config.maxFEs = 1000
# set config, indicate the agent and optimizer to train
config.train_agent = 'GLEET_Agent'
config.train_optimizer = 'GLEET_Optimizer'
config.train_epoch = get_epoch_dict(config.dataset)[config.train_agent[:-6]]
# [cont, comp, z-score]
config.fitness_mode = 'z-score'
config.in_task_agg = 'np.mean'
config.out_task_agg = 'np.mean'
config.use_ela = True
config.count_ela_fes = False    # ignore the extra FEs consumed when count_ela_fes set to False
# indicate training data and test data
trainer = Trainer(config, train_set, test_set, seed=0, fe=None, save_checkpoint = True)
result = trainer.train(trival=True)
print(config.train_agent)
print(result)
scores = calculate_per_task_perf(raw_data=result, fitness_mode=config.fitness_mode, cost_baseline=get_test_cost_baseline(config.dataset)[config.train_agent[:-len('_Agent')]])
print(scores)
print(config.use_ela, config.count_ela_fes, config.train_agent)