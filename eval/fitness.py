import numpy as np

def calculate_task_performance(results, cost_baseline):
    avg_result = {}
    for key in results.keys():
        avg_performance_gain = 0
        for i in range(len(results[key])):
            if cost_baseline[key][i] < 1e-8:
                b = 1e-8
            else:
                b = cost_baseline[key][i]
            if results[key][i] < 1e-8:
                a = 1e-8
            else:
                a = results[key][i]
            avg_performance_gain += (np.log10(a) - np.log10(b)) / (np.log10(b) + 9)
        avg_result[key] = avg_performance_gain/len(results[key])
    return avg_result

def calculate_compare_performance(results, cost_baseline):
    avg_result = {}
    for key in results.keys():
        avg_score = 0
        for sub_run in range(len(results[key])):
            sub_score = 0
            if (results[key][sub_run] <= cost_baseline[key][sub_run]) or (results[key][sub_run] <= 1e-8 and cost_baseline[key][sub_run] <= 1e-8):
                sub_score = -1
            avg_score += sub_score
        avg_result[key] = avg_score/len(results[key])
    return avg_result

def calculate_z_performance(results, cost_baseline):
    avg_result = {}
    for key in results.keys():
        mean_baseline = max(np.mean(cost_baseline[key]), 1e-8)
        sigma_baseline = max(np.std(cost_baseline[key]), 1e-8)
        avg_score = 0
        for sub_run in range(len(results[key])):
            # todo: may have bug
            if results[key][sub_run] <= 1e-8 and mean_baseline <= 1e-8:
                sub_score = 0
            else:
                sub_score = ((results[key][sub_run] - mean_baseline) / sigma_baseline)
            avg_score += sub_score
        avg_result[key] = avg_score/len(results[key])
    return avg_result

def calculate_per_task_perf(raw_data, fitness_mode, cost_baseline):
    # todo switch
    if fitness_mode == 'cont':
        return calculate_task_performance(raw_data, cost_baseline)
    elif fitness_mode == 'comp':
        return calculate_compare_performance(raw_data, cost_baseline)
    elif fitness_mode == 'z-score':
        return calculate_z_performance(raw_data, cost_baseline)

def calculate_aggregate_performance(task_performance_results, agent_list, in_task_agg, out_task_agg): # 3 dict {raw data->dict, task_perf->dict)}
    final_results = {}
    final_results['task_performance_results'] = task_performance_results
    final_score_list = []
    per_task_scores = {}

    # aggregate per-task scores
    for task_id in range(len(agent_list)):
        task_perf = task_performance_results[task_id]['task_perf']
        raw_data = task_performance_results[task_id]['raw_data']
        scores_list = []
        for key in task_perf.keys():
            scores_list.append(task_perf[key])
        # ! mean or median or max or sum
        per_score = eval(in_task_agg)(scores_list)
        per_task_scores['task-'+str(task_id)] = per_score

        final_score_list.append(per_score)
    # ! mean or max
    final_score = eval(out_task_agg)(final_score_list)
    final_results['per_task_scores'] = per_task_scores
    final_results['final_score'] = final_score
    return final_results