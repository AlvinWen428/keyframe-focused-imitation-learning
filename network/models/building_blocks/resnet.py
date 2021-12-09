import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels=3, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=None)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        if block.__name__ == 'Bottleneck':
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, [x0, x1, x2, x3, x4]  # output, intermediate

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers


def resnet18(pretrained=False, input_channels=3, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param input_channels: The channels of input image
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_channels, **kwargs)
    if pretrained:

        model_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_conv1 = model_dict['conv1.weight'].data.numpy()
        # remove the input layers if input the stack images
        if input_channels != 3:
            del model_dict['conv1.weight']
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)

        if input_channels != 3:
            # load the pretrained weights for conv1
            state_conv1 = state['conv1.weight'].data.numpy() * 0.1
            state_conv1[:, -3:, ...] = pretrained_conv1
            state['conv1.weight'] = torch.nn.parameter.Parameter(torch.Tensor(state_conv1))

        model.load_state_dict(state)
    return model


def resnet34(pretrained=False, input_channels=3, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param input_channels: The channels of input image
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_channels, **kwargs)
    if pretrained:

        model_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_conv1 = model_dict['conv1.weight'].data.numpy()
        # remove the input layers if input the stack images
        if input_channels != 3:
            del model_dict['conv1.weight']
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)

        if input_channels != 3:
            # load the pretrained weights for conv1
            state_conv1 = state['conv1.weight'].data.numpy() * 0.1
            state_conv1[:, -3:, ...] = pretrained_conv1
            state['conv1.weight'] = torch.nn.parameter.Parameter(torch.Tensor(state_conv1))

        model.load_state_dict(state)

    return model


def resnet50(pretrained=False, input_channels=3, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_channels, **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_conv1 = model_dict['conv1.weight'].data.numpy()
        # remove the input layers if input the stack images
        if input_channels != 3:
            del model_dict['conv1.weight']
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)

        if input_channels != 3:
            # load the pretrained weights for conv1
            state_conv1 = state['conv1.weight'].data.numpy() * 0.1
            state_conv1[:, -3:, ...] = pretrained_conv1
            state['conv1.weight'] = torch.nn.parameter.Parameter(torch.Tensor(state_conv1))

        model.load_state_dict(state)

    return model



def resnet101(pretrained=False, input_channels=3, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], input_channels, **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_conv1 = model_dict['conv1.weight'].data.numpy()
        # remove the input layers if input the stack images
        if input_channels != 3:
            del model_dict['conv1.weight']
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)

        if input_channels != 3:
            # load the pretrained weights for conv1
            state_conv1 = state['conv1.weight'].data.numpy() * 0.1
            state_conv1[:, -3:, ...] = pretrained_conv1
            state['conv1.weight'] = torch.nn.parameter.Parameter(torch.Tensor(state_conv1))

        model.load_state_dict(state)
    return model


def resnet152(pretrained=False, input_channels=3, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], input_channels, **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet152'])
        pretrained_conv1 = model_dict['conv1.weight'].data.numpy()
        # remove the input layers if input the stack images
        if input_channels != 3:
            del model_dict['conv1.weight']
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)

        if input_channels != 3:
            # load the pretrained weights for conv1
            state_conv1 = state['conv1.weight'].data.numpy() * 0.1
            state_conv1[:, -3:, ...] = pretrained_conv1
            state['conv1.weight'] = torch.nn.parameter.Parameter(torch.Tensor(state_conv1))

        model.load_state_dict(state)
    return model
