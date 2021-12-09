import argparse
import numpy as np
import torch

from action_correlation_model import train_ape_model


def get_prev_actions(index, img_path_list, prev_action_num, measurements_list):
    previous_actions_list = []
    previous_action_index = index
    previous_action_index_buffer = index
    while len(previous_actions_list) < prev_action_num * 3:
        previous_action_index -= 3
        if (previous_action_index < 0) or (
                img_path_list[previous_action_index].split('/')[0] != img_path_list[index].split('/')[0]):
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['brake'])
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['throttle'])
            previous_actions_list.append(measurements_list[previous_action_index_buffer]['steer'])
        else:
            previous_actions_list.append(measurements_list[previous_action_index]['brake'])
            previous_actions_list.append(measurements_list[previous_action_index]['throttle'])
            previous_actions_list.append(measurements_list[previous_action_index]['steer'])
            previous_action_index_buffer = previous_action_index
    previous_actions_list.reverse()
    previous_action_stack = np.array(previous_actions_list).astype(np.float)
    return previous_action_stack


def get_target_actions(index, img_path_list, target_action_num, measurements_list):
    target_actions_list = []
    target_actions_index = index
    target_actions_index_buff = index
    while len(target_actions_list) < target_action_num * 3:
        if (target_actions_index >= len(img_path_list)) or (img_path_list[target_actions_index].split('/')[0] != img_path_list[index].split('/')[0]):
            target_actions_list.append(measurements_list[target_actions_index_buff]['steer'])
            target_actions_list.append(measurements_list[target_actions_index_buff]['throttle'])
            target_actions_list.append(measurements_list[target_actions_index_buff]['brake'])
        else:
            target_actions_list.append(measurements_list[target_actions_index]['steer'])
            target_actions_list.append(measurements_list[target_actions_index]['throttle'])
            target_actions_list.append(measurements_list[target_actions_index]['brake'])
            target_actions_index_buff = target_actions_index
        target_actions_index += 3
    return np.array(target_actions_list).astype(np.float)


def main(data_path, prev_actions, curr_actions, model_layer_neurons, epoch_num):
    layer_num_str = [str(n) for n in model_layer_neurons]

    action_prediction_model = train_ape_model(data_path, prev_actions, curr_actions, epoch_num, model_layer_neurons, if_save=False)
    action_prediction_model.eval()

    img_path_list, measurements_list = np.load(data_path)
    for index in range(len(img_path_list)):
        previous_actions = get_prev_actions(index, img_path_list, prev_actions, measurements_list)
        current_action = get_target_actions(index, img_path_list, curr_actions, measurements_list)

        previous_actions = torch.FloatTensor(previous_actions).cuda().unsqueeze(0)
        current_action = torch.FloatTensor(current_action).cuda().unsqueeze(0)

        predict_curr_action = action_prediction_model(previous_actions)
        test_loss = ((predict_curr_action - current_action).pow(2) * torch.Tensor(
            [0.5, 0.45, 0.05]).cuda()).sum().cpu().item()
        measurements_list[index]['action_predict_loss'] = test_loss

    np.save(data_path.replace('.npy',
                              '_prev{}_curr{}_layer{}.npy'.format(prev_actions, curr_actions, '-'.join(layer_num_str))),
            [img_path_list, measurements_list])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./_preloads/100hours_CoILTrain100.npy',
                        help='the path of the preloaded .npy file in the _preload/ of the dataset')
    parser.add_argument('--prev-actions', type=int, default=9,
                        help='use how many previous actions (a_{t-1}, a_{t-2}, ...) to predict the current action.')
    parser.add_argument('--curr-actions', type=int, default=1,
                        help='use previous actions to predict how many next actions (a_{t}, a_{t+1}, ...).')
    parser.add_argument('--neurons', type=int, nargs='+', default=[300], help='the dimensions of the FC layers')
    parser.add_argument('--epoch', type=int, default=300, help='the training epochs')
    args = parser.parse_args()
    main(args.data,  args.prev_actions, args.curr_actions, args.neurons, args.epoch)

