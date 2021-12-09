from torch.nn import functional as F
import torch


def compute_branches_masks(controls, number_targets):
    """
        Args
            controls
            the control values that have the following structure
            command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go straight
            size of targets:
            How many targets is produced by the network so we can produce the masks properly
        Returns
            a mask to have the loss function applied
            only on over the correct branch.
    """

    """ A vector with a mask for each of the control branches"""
    controls_masks = []

    # when command = 2, branch 1 (follow lane) is activated
    controls_b1 = (controls == 2)
    controls_b1 = torch.tensor(controls_b1, dtype=torch.float32).cuda()
    controls_b1 = torch.cat([controls_b1] * number_targets, 1)
    controls_masks.append(controls_b1)
    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2] * number_targets, 1)
    controls_masks.append(controls_b2)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3] * number_targets, 1)
    controls_masks.append(controls_b3)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4] * number_targets, 1)
    controls_masks.append(controls_b4)

    return controls_masks


def l2_loss(params):
    """
        Functional LOSS L2
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):
        loss_branches_vec.append(((params['branches'][i] - params['targets']) **2
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append((params['branches'][-1] - params['inputs']) ** 2
                             * params['branch_weights'][-1])
    return loss_branches_vec, {}


def l1_loss(params):
    """
        Functional LOSS L1
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):
        loss_branches_vec.append(torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append(torch.abs(params['branches'][-1] - params['inputs'])
                             * params['branch_weights'][-1])
    return loss_branches_vec, {}

