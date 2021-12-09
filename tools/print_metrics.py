import json
import os
import argparse
import numpy as np
import pandas as pd


def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b


def add_not_start(metric_dict):
    metric_dict['not_start'] = {}
    path_completion = metric_dict['path_completion']
    for weather, value in path_completion.items():
        metric_dict['not_start'][weather] = []
        for task_result in value:
            metric_dict['not_start'][weather].append([1 if result < 0.01 else 0 for result in task_result])
    return metric_dict


def output(metric_dict, task_divide=False, weather_divide=True):
    # obtain the number of tasks
    for k, v in metric_dict.items():
        for weather, value in v.items():
            task_num = len(value)
            break
        break

    metric_dict = add_not_start(metric_dict)

    for k, v in metric_dict.items():
        exp_results = [[] for i in range(task_num)]
        for weather, value in v.items():
            for i in range(task_num):
                exp_results[i].append(value[i])
        exp_results = [flatten(r) for r in exp_results]  # flatten the list of each task's result
        # exp_results = flatten(exp_results)
        if task_divide:
            exp_results = [np.array(r) for r in exp_results]
            each_task_results = []
            for task_id in range(len(exp_results)):
                if (k != 'episodes_fully_completed') and ('collision' not in k) and (k != 'not_start') \
                        and (k != 'time_out'):
                    max = np.max(exp_results[task_id])
                    min = np.min(exp_results[task_id])
                    mean = np.mean(exp_results[task_id])
                    each_task_results.append(mean)
                    print(k, task_id, 'max:', max, 'min:', min, 'mean:', mean)
                else:
                    max = np.max(exp_results[task_id])
                    min = np.min(exp_results[task_id])
                    sum = np.sum(exp_results[task_id])
                    each_task_results.append(sum)
                    print(k, task_id, 'max:', max, 'min:', min, 'sum:', sum)
            print(k, 'eval mean', np.mean(each_task_results), 'eval std', np.std(each_task_results))
        else:
            exp_results = np.array(flatten(exp_results))
            if (k != 'episodes_fully_completed') and ('collision' not in k) and (k != 'not_start') \
                    and (k != 'time_out'):
                max = np.max(exp_results)
                min = np.min(exp_results)
                mean = np.mean(exp_results)
                print(k, 'max:', max, 'min:', min, 'mean:', mean)
            else:
                max = np.max(exp_results)
                min = np.min(exp_results)
                sum = np.sum(exp_results)
                print(k, 'max:', max, 'min:', min, 'sum:', sum)

        if weather_divide:
            print(k, end=' ')
            for weather, value in v.items():
                print(weather, end=': ')
                if (k != 'episodes_fully_completed') and ('collision' not in k) and (k != 'not_start') \
                        and (k != 'time_out'):
                    merged_data = [np.mean(data_list) for data_list in value]
                    print('mean:', np.mean(merged_data), 'std', np.std(merged_data), end='; ')
                else:
                    merged_data = [np.sum(data_list) for data_list in value]
                    print('mean:', np.mean(merged_data), 'std', np.std(merged_data), end='; ')
            print('\n')


def find_failure(summary):
    fail_exp_list = []
    for i in range(summary.shape[0]):
        if summary.loc[i, 'result'] == 0:
            fail_exp_list.append('episode_{}_{}_{}.{}'.format(summary.loc[i, 'weather'],
                                                              summary.loc[i, 'exp_id'],
                                                              summary.loc[i, 'start_point'],
                                                              summary.loc[i, 'end_point']))
    print('failure episodes:')
    print(fail_exp_list)


def merge_metrics(metrics_list):
    merged_results = metrics_list[0]
    for metrics in metrics_list[1:]:
        for measure_name, weather_results in metrics.items():
            for weather, task_results in weather_results.items():
                merged_results[measure_name][weather].extend(task_results)
    return merged_results


def main(args):
    if (args.path is None) or (args.path == 'None'):
        print('Please input valid path!')
    path_list = args.path

    metrics_list = []
    for p in path_list:
        path = os.path.join(p, 'metrics.json')
        summary_path = os.path.join(p, 'summary.csv')

        if not os.path.exists(path):
            print('The path does not exist!')
            return

        with open(path, 'r') as f:
            metrics_list.append(json.load(f))

    metrics = merge_metrics(metrics_list)

    # merge all the three collision type together
    metrics['end_collision'] = {}
    for weather in metrics['end_pedestrian_collision'].keys():
        metrics['end_collision'][weather] = np.sum([metrics['end_pedestrian_collision'][weather],
                                                    metrics['end_vehicle_collision'][weather],
                                                    metrics['end_other_collision'][weather]], axis=0).tolist()

    output(metrics, args.task_divide, args.weather_divide)

    # summary = pd.read_csv(summary_path)
    # find_failure(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str, nargs='+', help='The full path of metric results.')
    parser.add_argument('-td', '--task_divide', default=False, action='store_true', help='Whether print metrics for each task seperately.')
    parser.add_argument('-wd', '--weather_divide', default=False, action='store_true', help='Whether print metrics for each weather seperately.')
    args = parser.parse_args()
    main(args)
