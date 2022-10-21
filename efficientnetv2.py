import torch
import torch.nn as nn
import numpy as np
import yaml

# SiLU(Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # for old Pytorch Versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

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
            SiLU(),
        )   

# Last Conv
class ConvBN_1x1(nn.Sequential):
    def __init__(self, inp, out, kernel_size=1, stride=1):
        super(ConvBN_1x1, self).__init__(
            nn.Conv2d(inp, out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size//2,
                    bias=False),
            nn.BatchNorm2d(out),
            SiLU(),                        
        )

# Depthwise Conv
class Depthwise_conv(nn.Module):
    def __init__(self, hidden, stride, bias=False):
        super(Depthwise_conv, self).__init__()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3,
                stride=stride, padding=1, 
                groups=hidden, bias=bias)

        self.bn = nn.BatchNorm2d(hidden)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x) ) )

# Pointwise Conv
class Pointwise_conv(nn.Module):
    def __init__(self, inp, out, stride=1, bias=False):
        super(Pointwise_conv, self).__init__()

        self.conv = nn.Conv2d(inp, out, kernel_size=1,
                stride=stride, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x) ) )

# SE Layer
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, inp, out, stride, expand_ratio, not_fused):
        super(MBConv, self).__init__()
        assert stride in [1, 2], 'Stride should be equal either 1 or 2'

        hidden_dim = int(inp*expand_ratio)
        self.identity = stride==1 and inp == out
        
        if not_fused:
            self.conv = nn.Sequential(
                # Depthwise + Pointwise
                Pointwise_conv(inp, hidden_dim),
                Depthwise_conv(hidden_dim, stride),
                # SE layer
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out),
                )

        else:
                # Fused-MBConv
                self.conv = nn.Sequential(
                # Depthwise 
                Depthwise_conv(hidden_dim, stride),
                # SE layer
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out),
                )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=2, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = make_divisible(24 * width_mult, 8)
        self.features = [ConvBN_3x3(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, not_fuse in self.cfgs:
            output_channel = make_divisible(c * width_mult, 8)
            for i in range(n):
                self.features.append(block(input_channel, output_channel, s if i == 0 else 1, t, not_fuse))
                input_channel = output_channel

        output_channel = make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        print(output_channel)
        self.features = nn.Sequential(*self.features)
        print(self.features)
        self.conv = Pointwise_conv(input_channel, output_channel)
        # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        print('Fused Done!')
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, Fuse
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)

def efficientnetv2_test():
    net = effnetv2_s()
    output = net(torch.randn(4, 3, 224, 244))
    assert output.shape == (4, 2), 'Something went wrong...'
    print('Success!')


if __name__ == "__main__":
    efficientnetv2_test()


