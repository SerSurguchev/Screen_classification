import torch
import torch.nn as nn
import numpy as np
import yaml


def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# First Conv
class ConvBN_3x3(nn.Sequential):
    def __init__(self, inp, out, kernel_size=3, stride=2):
        super(ConvBN_3x3, self).__init__(
            nn.Conv2d(inp, out, 
                    kernel_size = kernel_size,
                    stride = stride, 
                    padding = kernel_size//2,
                    bias = False,),
            nn.BatchNorm2d(out),
            nn.ReLU6(inplace=True),
        )    

# Last Conv
class ConvBN_1x1(nn.Sequential):
    def __init__(self, inp, output):
        super(ConvBN_1x1, self).__init__(
            nn.Conv2d(inp, output, 
                    kernel_size=1,
                    stride=1, 
                    padding=0, bias=False),

            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True),
        )

# Depthwise Conv
class Depthwise_conv(nn.Module):
    def __init__(self, hidden, stride, bias=False):
        super(Depthwise_conv, self).__init__()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3,
                stride=stride, padding=1, 
                groups=hidden, bias=bias)

        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x) ) )

# Pointwise Conv
class Pointwise_conv(nn.Module):
    def __init__(self, inp, out, stride=1, bias=False):
        super(Pointwise_conv, self).__init__()

        self.conv = nn.Conv2d(inp, out, kernel_size=1,
                stride=stride, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x) ) )

# Inverted residual
class InvertedResidual(nn.Module):
    def __init__(self, inp, out, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], 'Stride should be equal either 1 or 2'

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == out

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise
                Depthwise_conv(hidden_dim, stride),
                # pw-linear
                nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out),
            )

        else:
            # Depthwise + Pointwise
            self.conv = nn.Sequential(
                Pointwise_conv(inp, hidden_dim),
                Depthwise_conv(hidden_dim, stride),
                # pw-linear
                nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, cfgs,  n_class=2, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.cfgs = cfgs

        assert input_size % 32 == 0
        self.features = [ConvBN_3x3(3, input_channel, 2)]
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_mult) if t > 1 else c

            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
               #  print('Done...')
                input_channel = output_channel

        # Building last several layers
        self.features.append(ConvBN_1x1(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        # Building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
