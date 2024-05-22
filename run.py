from eval.evaluator import eval_net_population
import numpy as np
from pypop7.optimizers.es import FCMAES, SEPCMAES, RMES, CMAES
import os
import time
from dataset.bbob import *
import ray
from config import get_config
from feature_extractor.feature_extractor import Feature_Extractor
from eval.utils import get_epoch_dict


def run_search():
    config = get_config()

    # Determine the testsuit used for top-level training, default to be BBOB
    config.dataset = 'bbob'

    if config.dataset == 'protein_docking':
        config.dim = 12
        config.maxFEs = 1000

    # grid complexity (You can change the model's hidden_dim and n_layers here)
    config.hidden_dim = 16
    config.n_layers = 1
    config.is_mlp = False  # Feature_extractor will be a pure MLP when config.is_mlp set to True

    # Determine the Population size of top-level optimizer 
    # It's calculated as 10 + max(log10(max(dim_of_featureExtractor - 3296, 1)))
    tmp_pe = Feature_Extractor(hidden_dim=config.hidden_dim, n_layers=config.n_layers, is_mlp=config.is_mlp)
    dim = tmp_pe.get_parameter_number()
    config.pop_size = 10 + int(np.ceil(np.log10(max(dim - 3296, 1))))
    print(f'Current pop_size is {config.pop_size}')

    # Determine the some hyper-parameters of top-level optimizer
    generations = 50
    problem = {'fitness_function': eval_net_population,
                    'ndim_problem': dim,
                    'upper_boundary':2 * np.ones(dim),
                    'lower_boundary':-2 * np.ones(dim)}
    # Adhere to grid-search options mentioned in our paper
    grid_options = [
        [np.zeros(dim), 0.1, 1], [np.zeros(dim), 0.1, 3],
        [np.zeros(dim), 0.3, 1], [np.zeros(dim), 0.3, 3], 
        [None, 0.1, 1], [None, 0.1, 3],
        [None, 0.3, 1], [None, 0.3, 3], 
    ]
    cur_grid = 4
    options = {'max_function_evaluations': config.pop_size * generations,  # set optimizer options
            'seed_rng': 1,
            'n_individuals': config.pop_size,
            'mean': grid_options[cur_grid][0],
            'sigma': grid_options[cur_grid][1],
            'c_multi': grid_options[cur_grid][2]}

    # Set up top-level optimizer
    es_options = ['FCMAES', 'SEPCMAES', 'RMES']
    es_choice = es_options[0]
    start = time.time()
    es = eval(es_choice)(problem,options)
    
    task_epoch_dict = get_epoch_dict(config.dataset)
    # Determine tasks used for training
    full_agent_list = list(task_epoch_dict.keys())
    cur_combination = [0, 2, 3]
    config.train_agent_list = [full_agent_list[i] for i in cur_combination]
    config.train_epoch_list = [task_epoch_dict[key] for key in config.train_agent_list]

    # choose fitness mode from [cont, comp, z-score], more details in eval/fitness.py
    config.fitness_mode = 'z-score'
    config.in_task_agg = 'np.mean'
    config.out_task_agg = 'np.mean'
    
    config.run_name = '_'.join(config.train_agent_list) + es_choice
    config.run_name = "{}_{}".format(config.run_name,time.strftime("%Y%m%dT%H%M%S"))
    config.log_dir = 'records/' + config.run_name + '/log_file/'
    config.save_dir = 'records/' + config.run_name + '/save_model/'
    mk_dir(config.log_dir)
    mk_dir(config.save_dir)
    
    results = es.optimize(args=(config,))
    end = time.time()
    print('best_value: {}'.format(results['best_so_far_y']))
    print('consumed time: {} seconds'.format(end-start))


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    ray.init()
    run_search()
