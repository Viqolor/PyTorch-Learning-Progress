import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_dataloaders(batch_size = 64):
    train_dataset = datasets.FashionMNIST(
        root = "data",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_dataset = datasets.FashionMNIST(
        root = "data",
        train = False,
        transform = ToTensor(),
        download = True
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader
