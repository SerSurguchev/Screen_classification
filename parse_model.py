import torch
from mobilenetv2 import MobileNetV2
from efficientnetv2 import EffNetV2


def BuildEffnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)

def BuildMobilenetv2(**kwargs):
    cfgs = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    return MobileNetV2(cfgs, **kwargs)   

def mobilenev2_test():
    net = BuildMobilenetv2()
    output = net(torch.randn(4, 3, 224, 224))
    assert output.shape == (4, 2), 'Something went wrong...'
    print('Success Mobilenet!')

def efficientnetv2_test():
    net = BuildEffnetv2_s()
    output = net(torch.randn(4, 3, 224, 244))
    assert output.shape == (4, 2), 'Something went wrong...'
    print('Success Efficientnet!')


if __name__ == "__main__":
    mobilenev2_test()
    efficientnetv2_test()
