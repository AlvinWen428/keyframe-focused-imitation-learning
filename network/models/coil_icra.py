from logger import coil_logger
import torch.nn as nn
import torch
import importlib
import os

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join


class CoILICRA(nn.Module):

    def __init__(self, params):
        # TODO: Improve the model autonaming function

        super(CoILICRA, self).__init__()
        self.params = params

        number_first_layer_channels = 0

        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.ALL_FRAMES_INCLUDING_BLANK

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        self.predicted_speed = 0

        if 'perception' in params:
            # For this case we check if the perception layer is of the type "conv"
            if 'conv' in params['perception']:
                avepool_output_size = params['perception']['conv']['avepool'] if 'avepool' in params['perception']['conv'] else None
                perception_convs = Conv(params={'channels': [number_first_layer_channels] +
                                                              params['perception']['conv']['channels'],
                                                'kernels': params['perception']['conv']['kernels'],
                                                'strides': params['perception']['conv']['strides'],
                                                'dropouts': params['perception']['conv']['dropouts'],
                                                'avepool': avepool_output_size,
                                                'end_layer': True})

                perception_fc = FC(params={'neurons': [perception_convs.get_conv_output(sensor_input_shape)]
                                                      + params['perception']['fc']['neurons'],
                                           'dropouts': params['perception']['fc']['dropouts'],
                                           'end_layer': False})

                self.perception = nn.Sequential(*[perception_convs, perception_fc])

                number_output_neurons = params['perception']['fc']['neurons'][-1]

            elif 'res' in params['perception']:  # pre defined residual networks
                resnet_module = importlib.import_module('network.models.building_blocks.resnet')
                resnet_module = getattr(resnet_module, params['perception']['res']['name'])
                self.perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                                input_channels=number_first_layer_channels,
                                                num_classes=params['perception']['res']['num_classes'])

                number_output_neurons = params['perception']['res']['num_classes']

            else:

                raise ValueError("invalid convolution layer type")

        if len(params['measurements']['fc']['neurons']) != 0:
            self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                       params['measurements']['fc']['neurons'],
                                           'dropouts': params['measurements']['fc']['dropouts'],
                                           'end_layer': False})
        else:
            self.measurements = None

        if 'previous_actions' in params:
            self.use_previous_actions = True
            self.previous_actions = FC(params={'neurons': [len(g_conf.TARGETS)*g_conf.NUMBER_PREVIOUS_ACTIONS] +
                                                          params['previous_actions']['fc']['neurons'],
                                               'dropouts': params['previous_actions']['fc']['dropouts'],
                                               'end_layer': False})
            number_preaction_neurons = params['previous_actions']['fc']['neurons'][-1]
        else:
            self.use_previous_actions = False
            number_preaction_neurons = 0

        if len(params['join']['fc']['neurons']) != 0:
            self.join = Join(
                params={'after_process':
                             FC(params={'neurons':
                                            [params['measurements']['fc']['neurons'][-1] +
                                             + number_preaction_neurons + number_output_neurons] +
                                            params['join']['fc']['neurons'],
                                         'dropouts': params['join']['fc']['dropouts'],
                                         'end_layer': False}),
                         'mode': 'cat'
                        }
             )
        else:
            self.join = None

        self.speed_branch = FC(params={'neurons': [number_output_neurons] +
                                                  params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})

        # Create the fc vector separatedely
        branch_fc_vector = []
        if len(params['join']['fc']['neurons']) != 0:
            for i in range(params['branches']['number_of_branches']):
                branch_fc_vector.append(FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                             params['branches']['fc']['neurons'] +
                                                             [len(g_conf.TARGETS)],
                                                   'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                                   'end_layer': True}))
        else:
            for i in range(params['branches']['number_of_branches']):
                branch_fc_vector.append(FC(params={'neurons': [params['perception']['fc']['neurons'][-1]] +
                                                             params['branches']['fc']['neurons'] +
                                                             [len(g_conf.TARGETS)],
                                                   'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                                   'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically

        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, a, pa=None):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x)
        ## Not a variable, just to store intermediate layers for future vizualization
        #self.intermediate_layers = inter

        """ ###### APPLY THE MEASUREMENT MODULE """
        if self.measurements is not None:
            m = self.measurements(a)
        else:
            m = None

        """ ###### APPLY THE PREVIOUS ACTIONS MODULE, IF THIS MODULE EXISTS"""
        if self.use_previous_actions and m is not None:
            n = self.previous_actions(pa)
            m = torch.cat((m, n), 1)

        """ Join measurements and perception"""
        if self.join is not None and m is not None:
            j = self.join(x, m)
        else:
            j = x

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(x)

        # We concatenate speed with the rest.
        return branch_outputs + [speed_branch_output]

    def forward_branch(self, x, a, branch_number, pa=None):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned
            pa: previous actions, optional

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        output = self.forward(x, a, pa)
        self.predicted_speed = output[-1]
        control = output[0:4]
        output_vec = torch.stack(control)

        return self.extract_branch(output_vec, branch_number)

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]

    def extract_predicted_speed(self):
        # return the speed predicted in forward_branch()
        return self.predicted_speed

