import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR, ReduceLROnPlateau
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
    if opt.model == 'MobileNetV2':
        model = BuildMobilenetv2()
    elif opt.model == 'EfficientNetV2':
        model = BuildEffnetv2_s()
    model = model.to(opt.device)

    # Optimizer
    if name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(opt.momentum, 0.999), weight_decay=opt.weight_decay)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=opt.momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=opt.momentum, nesterov=True)

    # Scheduler    
    if opt.scheduler:
        if opt.scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        elif opt.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                            factor=0.1, patience=10, 
                                            threshold=0.0001, threshold_mode='abs')
    else:
        scheduler = False

    # Criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    save_best_model = SaveBestModel()

    # Training Process
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    for epoch in range(opt.epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {opt.epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion, 
                                                  opt.device)

        valid_epoch_loss, valid_epoch_acc = validation(model, valid_loader,
                                                       criterion, opt.device)
        if scheduler:
            scheduler.step(epoch)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        save_best_model(
            valid_epoch_loss, model, optimizer,
            criterion, file_name_to_save
        )

        print('-' * 15, ' Epoch complete ', '-' * 15)        

    save_model(model, optimizer, criterion)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MobileNetV2', 
                        choices=['MobileNetV2', 'EfficientNetV2'], 
                        help='Choose classification model')

    parser.add_argument('--device', default='cuda', help='GPU device 0,1,2 or CPU')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                        choices=['SGD', 'Adam', 'AdamW'], 
                        help='optimizer')
        
    parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--scheduler', default=False, choices=['ReduceLROnPlateau','StepLR', False],
                        help='lr scheduler')

    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of epochs to train')  
    parser.add_argument('--transfer', action='store_true', help='Use pretrained model weighrs or not')
    parser.add_argument('--qat', action='store_true', help='Quantization aware training or not')          
    parser.add_argument('--sr', default=False, help='Train with channel sparsity regularization')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_args()


    



