import torch
import torch.optim as optim
import argparse

from utils import (
    save_model,
    SaveBestModel,
    seed_everything
)

from parse_model import (
    BuildEffnetv2_s,
    BuildMobilenetv2
)

from dataset import (
    create_dataset,
    train_loader,
    test_loader
)

from train import (
    train, 
    validation
)

seed_everything(42)

def main(opt):

    # Create data 

    # Initialize model

    # Optimizer

    # Loss function

    # Scheduler


    save_best_model = SaveBestModel()

    # Training Process

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MobileNetV2', choices=['MobileNetV2',
                                                                            'EfficientNetV2'], 
                        help='Choose classification model')

    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 
                                                                        'Adam', 
                                                                        'AdamW'], 
                        help='optimizer')        


    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()


    



