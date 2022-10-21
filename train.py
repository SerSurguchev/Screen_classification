import torch
from tqdm.auto import tqdm


def train(model, train_loader, optimizer, criterion, device=None):
    """
    Function with training body description
    Parameters:
    :param model: The model on which the data is trained
    :param train_loader: Batch of images
    :param optimizer: Model optimizer
    :param criterion: Loss function
    :param device: CUDA or CPU
    """
    model.train()
    print('Training...')

    train_running_loss = 0.0
    train_running_correct = 0.0
    counter = 0

    scaler = torch.cuda.amp.GradScaler()

    for i, data in enumerate(tqdm(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        
        # Zero grad
        optimizer.zero_grad()
       
        with torch.cuda.amp.autocast():
            outputs = model(image)
            
            loss = criterion(outputs, labels)

        train_running_loss += loss.item()
        _, preds = torch.max(output, 1)
        train_running_correct += (preds==labels).sum().item()
        
        # Backpropagation
        scaler.scale(loss).backward()

        # Update the optimizer parameters
        scaler.step(optimizer)
        scaler.update()

    # Loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validation(model, val_loader, criterion):
    """
    Function with validation body description
    Parameters:
    :param model: The model on which the data is trained
    :param val_loader: Batch of images
    :param criterion: Loss function
    """
    model.eval()
    print('Validation...')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            counter += 1

            image, labels = data
            image = image.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(image)

                # Calculate the loss
                loss = criterion(outputs, labels)

            valid_running_loss += loss.item()

            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(val_loader.dataset))
    return epoch_loss, epoch_acc
