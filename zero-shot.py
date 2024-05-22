import torch
import copy
from config import get_config
from trainer import Trainer
from feature_extractor.feature_extractor import Feature_Extractor
import pickle
from dataset.bbob import *
from dataset.protein_docking import *
from eval.fitness import *
import ray
from eval.cost_baseline import get_test_cost_baseline
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



def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



@ray.remote(num_cpus=1, num_gpus=0)
def single_task(config, task_id, trainset, testset, seed, cost_baseline, fe):

    config.train_agent = config.test_agent_list[task_id] + '_Agent'
    config.train_optimizer = config.test_agent_list[task_id] + '_Optimizer'
    config.train_epoch = config.test_epoch_list[task_id]

    trainer = Trainer(config, trainset, testset, seed, fe=fe)
    results = trainer.train(pick_best=True)
    
    return {'raw_data': results, 'task_perf': calculate_per_task_perf(results, fitness_mode=config.fitness_mode, cost_baseline=cost_baseline)}


def evaluate(fe,train_set, test_set,config):
    object_refs = [single_task.remote(copy.deepcopy(config), tid, copy.deepcopy(train_set), copy.deepcopy(test_set), 0, copy.deepcopy(get_test_cost_baseline(config.dataset)[config.test_agent_list[tid]]), copy.deepcopy(fe)) for tid in range(len(config.test_agent_list))]
    results = calculate_aggregate_performance(ray.get(object_refs), config.test_agent_list, config.in_task_agg, config.out_task_agg)
    return results



if __name__ == '__main__':
    ray.init()
    seed = 0
    load_path = '/home/chenjiacheng/Auto-feature/records/LDE_RLEPSO_RL_DAS_z-score_np.mean_20240429T015706/save_model/20240429T152953.pkl'
    
    fe = load_data(load_path)
    config = get_config()
    config.hidden_dim = 16
    config.n_layers = 1
    config.is_mlp = False
    feature_embedder = vector2nn(fe, Feature_Extractor(hidden_dim=config.hidden_dim, n_layers=config.n_layers, is_mlp=config.is_mlp))

    
    config.dataset = 'bbob'
    train_set, test_set = construct_problem_set(dataset=config.dataset)


    if config.dataset == 'protein_docking':
        config.dim = 12
        config.maxFEs = 1000

    task_epoch_dict = get_epoch_dict(config.dataset)
    
    combinations = [[0, 2, 3]]
    test_combinations = [list(set([i for i in range(len(task_epoch_dict.keys()))]) - set(combinations[k])) for k in range(len(combinations))]
    # test_combinations = combinations
    print(test_combinations)
    full_agent_list = list(task_epoch_dict.keys())
    cur_combination = test_combinations[-1]
    config.test_agent_list = [full_agent_list[i] for i in cur_combination]
    config.test_epoch_list = [task_epoch_dict[key] for key in config.test_agent_list]

    config.fitness_mode = 'z-score'
    config.in_task_agg = 'np.mean'
    config.out_task_agg = 'np.mean'


    # {'task_perf', 'per_task_scores', 'final_score'}
    results = evaluate(feature_embedder, train_set, test_set, config)

    print(results.keys())
    print(f"********** Per task score {config.dataset} {config.fitness_mode} {config.in_task_agg} ************")
    print(config.test_agent_list)
    print(results['per_task_scores'])
    print(results['final_score'])
    for tid, ag in enumerate(config.test_agent_list):
        print(f"ag: {ag}, task_perf:{results['task_performance_results'][tid]}")
    base_path = '/'.join(load_path.split('/')[:-2])
    run_name = load_path.split('/')[-1][:-4]
    with open(base_path + f'/reload_fe_{run_name}_{config.dataset}_pickbest.pkl', 'wb') as f:
        pickle.dump(results, f, -1)