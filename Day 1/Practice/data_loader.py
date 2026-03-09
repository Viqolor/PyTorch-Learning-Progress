import math
import sys
import pandas as pd
import numpy as np

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_dataloaders(batch_size = 64):
    train_dataset = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader