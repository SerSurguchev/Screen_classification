import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import numpy as np
import math

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
                    padding=0, 
                    bias=False),

            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2], 'Stride should be equal either 1 or 2'

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise conv
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise conv
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise conv
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, cfgs,  n_class=2, input_size=224, QAT=None,  width_mult=1., dropout=0.2):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        self.dropout = dropout
        self.cfgs = cfgs
        # Quantization aware training or not    
        self.QAT = QAT
        # QuantStub converts tensors from floating point to quantized
        self.quant = QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = DeQuantStub()

        assert input_size % 32 == 0
        self.features = [ConvBN_3x3(3, input_channel, 3)]
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else 1280

        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        self.features.append(ConvBN_1x1(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        # Building classifier
        self.classifier = nn.Sequential(
                        nn.Dropout(self.dropout),
                        nn.Linear(self.last_channel, n_class),
                        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBN_1x1 or type(m) == ConvBN_3x3:
                fuse_modules(m, [['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

    def forward(self, x):
        if self.QAT:
            x = self.quant(x)
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            x = self.dequant(x)
        else:
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x
