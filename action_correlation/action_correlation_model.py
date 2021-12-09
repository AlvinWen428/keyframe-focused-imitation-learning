import os
from functools import partial
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


class ActionModel(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=None):
        super(ActionModel, self).__init__()
        self.units = neurons if neurons is not None else [300]

        layer_list = [nn.Linear(input_dim, self.units[0]), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(len(self.units) - 1):
            layer_list.append(nn.Linear(self.units[i], self.units[i+1]))
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        layer_list.append(nn.Linear(self.units[-1], output_dim))

        self.main = nn.Sequential(*layer_list)
        self.max_loss, self.min_loss = 0, 0

    def forward(self, input):
        output = self.main(input)
        return output

    def load_margin(self, min_loss, max_loss):
        self.min_loss = min_loss
        self.max_loss = max_loss


class ActionDataset(Dataset):
    def __init__(self, path, prev_actions, curr_actions):
        self.path = path
        self.prev_actions, self.curr_actions = prev_actions, curr_actions
        self.stack_size = self.prev_actions + self.curr_actions
        self.actions = []
        self.all_steers, self.all_throttles, self.all_brakes = [], [], []

        img_path_list, measurements = np.load(path)
        for index in range(len(img_path_list)):
            start_index = index - prev_actions * 3
            end_index = index + (curr_actions - 1) * 3
            if start_index >= 0 and img_path_list[start_index].split('/')[0] == img_path_list[index].split('/')[0]:
                if end_index < len(img_path_list) and img_path_list[end_index].split('/')[0] == img_path_list[index].split('/')[0]:
                    action_list = []
                    for idx in range(start_index, end_index+1, 3):
                        action_list.append(measurements[idx]['steer'])
                        action_list.append(measurements[idx]['throttle'])
                        action_list.append(measurements[idx]['brake'])
                    self.actions.append(np.array(action_list))
                    self.all_steers.append(measurements[index]['steer'])
                    self.all_throttles.append(measurements[index]['throttle'])
                    self.all_brakes.append(measurements[index]['brake'])

        self.mean = np.array([np.mean(self.all_steers), np.mean(self.all_throttles), np.mean(self.all_brakes)])
        self.std = np.array([np.std(self.all_steers), np.std(self.all_throttles), np.std(self.all_brakes)])
        self.stacked_mean = np.tile(self.mean, self.stack_size)
        self.stacked_std = np.tile(self.std, self.stack_size)

        for index in range(len(self.actions)):
            stacked_actions = self.actions[index]
            self.actions[index] = torch.FloatTensor(stacked_actions)

    def __getitem__(self, idx):
        return self.actions[idx][:-3*self.curr_actions], self.actions[idx][-3*self.curr_actions:]

    def __len__(self):
        return len(self.actions)

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std


def train(model, optimizer, train_loader, loss_func, epoch, logger):
    model.train()
    optimizer.zero_grad()
    accumulate_loss = []
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        model.zero_grad()
        output = model(batch_x)
        loss = loss_func(output, batch_y).mean()

        accumulate_loss.append(float(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    logger.add_scalar('train_loss', np.mean(accumulate_loss), epoch)
    print('iter: {} train loss: {}'.format(epoch, np.mean(accumulate_loss)))


def test(model, test_loader, loss_func, epoch, logger):
    model.eval()
    accumulate_loss = []
    min_loss, max_loss = np.inf, 0
    with torch.no_grad():
        for batch_idx, (test_x, test_y) in enumerate(test_loader):
            test_x, test_y = test_x.cuda(), test_y.cuda()

            output = model(test_x)
            loss = loss_func(output, test_y)

            max_value = loss.max().item()
            min_value = loss.min().item()
            if min_value < min_loss:
                min_loss = min_value
            if max_value > max_loss:
                max_loss = max_value

            loss = loss.mean()
            accumulate_loss.append(float(loss.item()))
    logger.add_scalar('test_loss', np.mean(accumulate_loss), epoch)
    print('iter: {} test loss: {}'.format(epoch, np.mean(accumulate_loss)))
    return min_loss, max_loss


def weighted_loss(output, target, weight=None):
    loss = (output - target).pow(2)
    if weight is not None:
        loss = loss * weight
        return loss.sum(dim=-1)
    else:
        return loss.mean(dim=-1)


def adjustlr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


def train_ape_model(data_path, prev_actions, curr_actions, epoch_num, model_layer_neurons, if_save):
    data_name = os.path.basename(data_path).split('.')[0]
    identifier = str(int(datetime.utcnow().timestamp())) + ''.join([str(np.random.randint(10)) for _ in range(8)])

    dataset = ActionDataset(data_path, prev_actions, curr_actions)
    mean, std = dataset.get_mean(), dataset.get_std()

    prev_seed = torch.get_rng_state()
    torch.manual_seed(0)
    trainset, testset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    torch.set_rng_state(prev_seed)

    trainloader = DataLoader(dataset=trainset, batch_size=512, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=512, shuffle=False)

    model = ActionModel(input_dim=prev_actions*3, output_dim=curr_actions*3, neurons=model_layer_neurons)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loss_func = partial(weighted_loss, weight=torch.Tensor([0.5, 0.45, 0.05]*curr_actions).cuda())

    layer_num_str = [str(n) for n in model_layer_neurons]
    result_dir = 'action_correlation/results/{}-prev{}-curr{}-layer{}-{}'.format(data_name, prev_actions, curr_actions, '-'.join(layer_num_str), identifier)
    save_dir = 'action_correlation/checkpoints/prev{}-curr{}-layer{}.pkl'.format(prev_actions, curr_actions, '-'.join(layer_num_str))

    log_dir = os.path.join(result_dir, 'run')
    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    min_loss, max_loss = 0, 0
    for epoch in range(1, epoch_num+1):
        train(model, optimizer, trainloader, loss_func, epoch, writer)
        min_loss, max_loss = test(model, testloader, loss_func, epoch, writer)

        if epoch % (epoch_num // 3) == 0:
            adjustlr(optimizer, 0.1)

    if if_save:
        os.makedirs('action_correlation/checkpoints', exist_ok=True)
        checkpoint = {'state_dict': model.state_dict(), 'min_loss': min_loss, 'max_loss': max_loss, 'mean': mean, 'std': std}
        torch.save(checkpoint, save_dir)

    return model
