import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size = 64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root = "data",
        train = True,
        transform = train_transform,
        download = True
    )
    test_dataset = datasets.FashionMNIST(
        root = "data",
        train = False,
        transform = test_transform,
        download = True
    )

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_dataloader, test_dataloader