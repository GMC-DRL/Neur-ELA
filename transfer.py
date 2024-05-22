import torch
import copy
from config import get_config
from trainer_finetune import Trainer
from feature_extractor.feature_extractor import Feature_Extractor
import pickle
from dataset.bbob import *
from eval.fitness import calculate_per_task_perf
from eval.cost_baseline import get_test_cost_baseline
import os

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



def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def single_task(config, task_id, trainset, testset, seed, cost_baseline, fe):

    config.train_agent = config.test_agent_list[task_id] + '_Agent'
    config.train_optimizer = config.test_agent_list[task_id] + '_Optimizer'
    config.train_epoch = config.test_epoch_list[task_id]

    trainer = Trainer(config, trainset, testset, seed, fe=fe)
    results = trainer.train(pick_best=False)
    
    return {'raw_data': results, 'task_perf': calculate_per_task_perf(results, fitness_mode=config.fitness_mode, cost_baseline=cost_baseline)}


combinations = [[0, 1, 2], [2, 5, 6], [2, 3, 5]]
test_combinations = [list(set([i for i in range(7)]) - set(combinations[k])) for k in range(3)]
print(test_combinations)


if __name__ == '__main__':

    seed = 0

    load_path = '/home/chenjiacheng/Auto-feature/records/LDE_RLEPSO_RL_DAS_z-score_np.mean_20240429T163107/save_model/20240429T183257.pkl'
    fe = load_data(load_path)

    feature_embedder = vector2nn(fe, Feature_Extractor())
    feature_embedder.set_on_train()

    config = get_config()
    config.dataset = 'bbob-noisy'
    train_set, test_set = construct_problem_set(dataset=config.dataset)


    if config.dataset == 'protein_docking':
        config.dim = 12
        config.maxFEs = 1000

    config.save_checkpoint_dir = 'debug'
    if not os.path.exists(config.save_checkpoint_dir):
        os.mkdir(config.save_checkpoint_dir)

    config.fitness_mode = 'z-score'
    config.in_task_agg = 'np.mean'
    config.out_task_agg = 'np.mean'

    full_agent_list = list(task_epoch_dict.keys())
    cur_combination = test_combinations[0]
    cur_combination = [7]
    config.test_agent_list = [full_agent_list[i] for i in cur_combination]
    config.test_epoch_list = [task_epoch_dict[key] for key in config.test_agent_list]
    

    collect_list = []
    for task_id in range(len(cur_combination)):
        result = single_task(config, task_id, trainset=train_set, testset=test_set, seed=seed, cost_baseline=get_test_cost_baseline(config.dataset)[config.test_agent_list[task_id]], fe=feature_embedder)
        collect_list.append(result)
        print(f"agent: {config.test_agent_list[task_id]},\n raw_data: {result['raw_data']}, \n task_perf:{result['task_perf']}")
        
    # for task_id in range(len(cur_combination)):
    #     print(f"agent: {config.test_agent_list[task_id]},\n raw_data: {result['raw_data']}, \n task_perf:{result['task_perf']}")
