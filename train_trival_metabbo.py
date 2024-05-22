import torch
import copy
from config import get_config
from trainer import Trainer
from feature_extractor.feature_extractor import Feature_Extractor
import pickle
from dataset.bbob import *
from eval.utils import get_epoch_dict, construct_problem_set

def vector2nn(x, net):
    assert len(x) == sum([param.nelement() for param in net.parameters()]), 'dim of x and net not match!'
    x_copy = copy.deepcopy(x)
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.FloatTensor(x_copy[ptr: ptr+num_of_params])
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net



seed = 0
config = get_config()
config.dataset = 'bbob'
train_set, test_set = construct_problem_set(dataset=config.dataset)


if config.dataset == 'protein_docking':
    config.dim = 12
    config.maxFEs = 1000

task_epoch_dict = get_epoch_dict(config.dataset)

# set config, indicate the agent and optimizer to train
config.train_agent = 'RLEPSO_Agent'
config.train_optimizer = 'RLEPSO_Optimizer'
config.train_epoch = task_epoch_dict[config.train_agent[:-6]]
# [cont, comp, z-score]
config.fitness_mode = 'z-score'
config.in_task_agg = 'np.mean'
config.out_task_agg = 'np.mean'
# indicate training data and test data
trainer = Trainer(config, train_set, test_set, seed)
result = trainer.train(trival=True)
print(config.train_agent)
print(result)
