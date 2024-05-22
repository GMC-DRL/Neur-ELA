import ray
import time
import numpy as np
from feature_extractor.feature_extractor import Feature_Extractor
from config import get_config
from trainer import Trainer
import pickle
import torch
import copy
from dataset.bbob import *
from eval.cost_baseline import get_train_cost_baseline
from eval.fitness import *
import os
import matplotlib.pyplot as plt

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

# load BBOB testsuits for training
def construct_problem_set():
    with open('dataset/trainset_v2.pkl', 'rb') as f:
        trainset = pickle.load(f)
    with open('dataset/testset_v2.pkl', 'rb') as f:
        testset = pickle.load(f)
    return trainset,testset


@ray.remote(num_cpus=1, num_gpus=0)
def single_task(feature_net, task_id, cost_baseline, trainset, testset, config, seed):
    # transform the vector into a neural network
    feature_embedder = vector2nn(feature_net, Feature_Extractor(hidden_dim=config.hidden_dim, n_layers=config.n_layers, is_mlp=config.is_mlp))
    print(' feature_embedder')
    

    # determine by config what the task is
    config.train_agent = config.train_agent_list[task_id] + '_Agent'
    config.train_optimizer = config.train_agent_list[task_id] + '_Optimizer'
    config.train_epoch = config.train_epoch_list[task_id]
    trainer = Trainer(config, trainset, testset, seed, feature_embedder)
    results = trainer.train(pick_best=True)
    
    return {'raw_data': results, 'task_perf': calculate_per_task_perf(results, fitness_mode=config.fitness_mode, cost_baseline=cost_baseline)}




@ray.remote(num_cpus=1, num_gpus=0)
def evaluate(feature_net,train_set, test_set,config):
    object_refs = [single_task.remote(feature_net, tid, copy.deepcopy(get_train_cost_baseline(config.dataset)[config.train_agent_list[tid]]), copy.deepcopy(train_set), copy.deepcopy(test_set), copy.deepcopy(config), 0) for tid in range(3)]
    results = calculate_aggregate_performance(ray.get(object_refs), config.train_agent_list, config.in_task_agg, config.out_task_agg)
    return results

def eval_net_population(feature_nets, config):
    train_set, test_set = construct_problem_set()
    object_refs = [evaluate.remote(feature_net,copy.deepcopy(train_set),copy.deepcopy(test_set), copy.deepcopy(config)) for feature_net in feature_nets]
    results = ray.get(object_refs) # 10 dicts, {'task_perf', 'per_task_scores', 'final_score'}
    # save the metadata
    run_time = f'{time.strftime("%Y%m%dT%H%M%S")}'
    with open(f'{config.log_dir}{run_time}.pkl', 'wb') as f:
        pickle.dump(results, f)
    # return to FCMAES only the final scores
    final_scores = [result['final_score'] for result in results]
    print(final_scores)

    # save model
    best_idx = np.argmin(final_scores)
    with open(f'{config.save_dir}{run_time}.pkl', 'wb') as f:
        pickle.dump(feature_nets[best_idx], f, -1)

    # plot todo: fix
    # plot_train_curve(path=config.log_dir, run_name=config.run_name)

    return final_scores

# def plot_train_curve(path, run_name):
    
#     files = os.listdir(path)
#     best_data = []
#     mean_data = []
#     for file in files:
#         # print(file)
#         with open(path + file, 'rb') as f:
#             res = pickle.load(f)
#             # print(res[0].keys())
#             # print(res[0]['per_task_scores'])
#         best_data.append(np.min([re['final_score'] for re in res]))
#         mean_data.append(np.mean([re['final_score'] for re in res]))
#         best_idx = np.argmin([re['final_score'] for re in res])
#         # print(res[best_idx]['per_task_scores'])
#     plt.clf()
#     plt.title(f'{run_name}_best')
#     plt.plot(np.arange(1,len(best_data)+1), best_data, 'r')
#     # plt.plot(np.arange(1,len(mean_data)+1), mean_data, 'b')
#     plt.savefig(f'{path}../best_evolution.png')
    
#     plt.clf()
#     plt.title(f'{run_name}_mean')
#     # plt.plot(np.arange(1,len(best_data)+1), best_data, 'r')
#     plt.plot(np.arange(1,len(mean_data)+1), mean_data, 'b')
#     plt.savefig(f'{path}../mean_evolution.png')