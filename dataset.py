from PIL import Image
import torch
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data

class CreateDataset(torch.utils.data.Dataset):
    """
    Class to create custom dataset
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        rgb_im = image.convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))
        return (rgb_im, y_label)

def train_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    '''
    Parameters:
    :param path: 
    :param batch_size: 
    :param num_workers: 
    :param pin_memory: 
    :return: 
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return data.DataLoader(
        datasets.ImageFoler(path,
                            transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x[np.random.permutation(3), :, :]),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    '''
    Parameters:
    :param path: 
    :param batch_size: 
    :param num_workers: 
    :param pin_memory: 
    :return: 
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return data.Dataloader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 normalize
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
