import os.path as osp

import torch
from torch.utils.data import DataLoader

from dataset import PMDDataset


def make_PMD_loader(dataset_path: str, seed: int, batch_size: int, scale: int) -> tuple:
    
    train_dir = osp.join(dataset_path, 'train')
    test_dir = osp.join(dataset_path, 'test', '_ALL')

    
    train_dataset = PMDDataset(train_dir, scale)
    train_size, val_size = getTrainTestCounts(train_dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    test_dataset = PMDDataset(test_dir, scale)
    
    print("Train dataset size: ", train_size)
    print("Validation dataset size: ", val_size)
    print("Test dataset size: ", test_dataset.__len__())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader
    
def getTrainTestCounts(dataset):
    train_size = int(dataset.__len__() * 0.95) 
    val_size   = dataset.__len__() - train_size
    return train_size, val_size