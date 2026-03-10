import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_loaders(batch_size = 64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    train_dataset = datasets.FashionMNIST(
        root = 'data',
        train = True,
        transform = train_transform,
        download = True
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    test_dataset = datasets.FashionMNIST(
        root = 'data',
        train = False,
        transform = test_transform,
        download = True
    )

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader