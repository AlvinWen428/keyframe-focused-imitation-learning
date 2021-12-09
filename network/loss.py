from . import loss_functional as LF
import torch
import torch.nn as nn

from action_correlation.action_correlation_model import ActionModel


def l1(params):
    return branched_loss(LF.l1_loss, params)


def l2(params):
    return branched_loss(LF.l2_loss, params)


def branched_loss(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])
    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to  have 4 branches not using speed.

    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['Steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['Gas'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['Brake']

    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    speed_loss = loss_branches_vec[4]/(params['branches'][0].shape[0])

    """ importance sampling """
    importance_sampling_method = params['importance_sampling_method']

    if importance_sampling_method == 'mean':
        weighted_loss = torch.sum(loss_function) / (params['branches'][0].shape[0]) \
               + torch.sum(speed_loss) / (params['branches'][0].shape[0])
        loss_info = {'unweighted_loss': loss_function}
    else:
        weight_importance_sampling = params['action_predict_loss']

        if importance_sampling_method == 'softmax':
            weighted_loss_function = loss_function * nn.functional.softmax(weight_importance_sampling / params['importance_sampling_softmax_temper'], dim=0)
        elif importance_sampling_method == 'threshold':
            scaled_weight_importance = (weight_importance_sampling > params['importance_sampling_threshold']).type(torch.float) * (params['importance_sampling_threshold_weight']-1) + 1
            weighted_loss_function = loss_function * scaled_weight_importance
        else:
            raise ValueError
        weighted_loss = torch.sum(weighted_loss_function) + torch.sum(speed_loss) / (params['branches'][0].shape[0])

        hard_samples_loss = loss_function[weight_importance_sampling > params['importance_sampling_threshold']]
        easy_samples_loss = loss_function[weight_importance_sampling <= params['importance_sampling_threshold']]
        loss_info = {'unweighted_loss': loss_function, 'hard_samples_loss': hard_samples_loss, 'easy_samples_loss': easy_samples_loss}
    return weighted_loss, loss_info, plotable_params


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == 'L1':

        return l1

    elif loss_name == 'L2':

        return l2

    else:
        raise ValueError(" Not found Loss name")


