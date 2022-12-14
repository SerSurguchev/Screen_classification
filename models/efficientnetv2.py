import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, not_fuse):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if not_fuse:
            self.conv = nn.Sequential(
                # Pointwise
                nn.Conv2d(inp, 
                        hidden_dim, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False),

                nn.BatchNorm2d(hidden_dim),
                SiLU(),

                # Depthwise
                nn.Conv2d(hidden_dim, 
                        hidden_dim, 
                        kernel_size=3, 
                        stride=stride, 
                        padding=1, 
                        groups=hidden_dim, 
                        bias=False),
               
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # SE layer
                SELayer(inp, hidden_dim),

                # Pw-linear
                nn.Conv2d(hidden_dim, 
                        oup, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False),
                
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Fused MBConv
                nn.Conv2d(inp, 
                        hidden_dim, 
                        kernel_size=3, 
                        stride=stride, 
                        padding=1, 
                        bias=False),

                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # SE layer
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
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
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [ConvBN_3x3(3, input_channel, 3)]

        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel

        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        layers.append(ConvBN_1x1(input_channel, output_channel))

        self.features = nn.Sequential(*layers)

        # Building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
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
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

