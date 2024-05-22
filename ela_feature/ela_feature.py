from pflacco.classical_ela_features \
import calculate_ela_meta, calculate_ela_conv, calculate_ela_level,\
    calculate_ela_curvate,calculate_ela_local,calculate_ela_distribution
import math
import numpy as np

def get_ela_feature(problem, Xs, Ys):
    total_calculation_time_cost = 0
    total_calculation_fes = 0
    all_features = []
    # meta features
    ela_meta_full_results = calculate_ela_meta(Xs,Ys)
    total_calculation_time_cost += ela_meta_full_results['ela_meta.costs_runtime']
    #print('meta features time cost: {},  consumed fes: {}'.format(ela_meta_full_results['ela_meta.costs_runtime'], 0))
    for k in ela_meta_full_results.keys():
        if k != 'ela_meta.costs_runtime':
            v = ela_meta_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)

    # convexity features
    ela_conv_full_results = calculate_ela_conv(Xs, Ys, problem.eval, ela_conv_nsample= 200)
    total_calculation_fes += ela_conv_full_results['ela_conv.additional_function_eval']
    total_calculation_time_cost += ela_conv_full_results['ela_conv.costs_runtime']
    #print('convexity features time cost: {},  consumed fes: {}'.format(ela_conv_full_results['ela_conv.costs_runtime'], ela_conv_full_results['ela_conv.additional_function_eval']))
    for k in ela_conv_full_results.keys():
        if (k != 'ela_conv.additional_function_eval') and (k != 'ela_conv.costs_runtime'):
            v = ela_conv_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)

    # level-set features
    try:
        ela_level_full_results = calculate_ela_level(Xs,Ys, ela_level_quantiles=[0.25, 0.5, 0.75], ela_level_resample_iterations=10)
    except:
        ela_level_full_results = {}
        for _k in range(9):
            ela_level_full_results[_k] = -1
        ela_level_full_results['ela_level.costs_runtime'] = 0
    total_calculation_time_cost += ela_level_full_results['ela_level.costs_runtime']
    #print('level-set features time cost: {},  consumed fes: {}'.format(ela_level_full_results['ela_level.costs_runtime'], 0))
    for k in ela_level_full_results.keys():
        if k != 'ela_level.costs_runtime':
            v = ela_level_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)


    # curvature features
    # ela_cur_full_results = calculate_ela_curvate(Xs, Ys, problem.eval, problem.dim, problem.lb, problem.ub, sample_size_factor = 1)
    # total_calculation_fes += ela_cur_full_results['ela_curv.costs_fun_evals:']
    # total_calculation_time_cost += ela_cur_full_results['ela_curv.costs_runtime']
    # print('curvature features time cost: {},  consumed fes: {}'.format(ela_cur_full_results['ela_curv.costs_runtime'], ela_cur_full_results['ela_curv.costs_fun_evals:']))
    # for k in ela_cur_full_results.keys():
    #     if (k != 'ela_curv.costs_fun_evals:') and (k != 'ela_curv.costs_runtime'):
    #         all_features.append(ela_cur_full_results[k])

    # local features
    ela_local_full_results = calculate_ela_local(Xs, Ys, problem.eval, problem.dim, problem.lb, problem.ub,ela_local_local_searches_factor=1)
    total_calculation_fes += ela_local_full_results['ela_local.additional_function_eval']
    total_calculation_time_cost += ela_local_full_results['ela_local.costs_runtime']
    #print('local features time cost: {},  consumed fes: {}'.format(ela_local_full_results['ela_local.costs_runtime'],
                                                                  #ela_local_full_results['ela_local.additional_function_eval']))
    for i,k in enumerate(ela_local_full_results.keys()):
        if i <= 6:
            v = ela_local_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)

    # distributional features
    ela_dis_full_results = calculate_ela_distribution(Xs,Ys)
    total_calculation_time_cost += ela_dis_full_results['ela_distr.costs_runtime']
    #print('distributional features time cost: {},  consumed fes: {}'.format(ela_dis_full_results['ela_distr.costs_runtime'], 0))
    for k in ela_dis_full_results.keys():
        if k != 'ela_distr.costs_runtime':
            v = ela_dis_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)

    return np.array(all_features), total_calculation_fes, total_calculation_time_cost



if __name__ == '__main__':
    from bbob import *
    train_set, test_set = BBOB_Dataset.get_datasets(suit='bbob',
                                                    dim=10,
                                                    upperbound=5,
                                                    shifted=False,
                                                    rotated=False,
                                                    biased=False)
    p = train_set[0]
    Xs = -5 + 10 * np.random.rand(100,10)
    Ys = p.eval(Xs)
    features = get_ela_feature(p,Xs,Ys)
    print(features)


