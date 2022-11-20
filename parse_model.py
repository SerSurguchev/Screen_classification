import torch
import torch.nn as nn
from torchsummary import summary
from mobilenetv2 import *
from efficientnetv2 import *


def BuildEffnetv2_s(pretrained=False, **kwargs):
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
    model = EffNetV2(cfgs, **kwargs)

    return model

def BuildMobilenetv2(pretrained=True, **kwargs):
    """
    Constructs a MobilenetV2 model 
    """
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

    model =  MobileNetV2(cfgs, **kwargs)   
    summary(model, (3, 224, 224))

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        pretrained_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', 
             progress=True)

        model_dict = model.state_dict()
        pretrained_dict_conv = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    return model

def mobilenev2_test():
    model = BuildMobilenetv2()
    output = model(torch.randn(4, 3, 224, 224))
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
