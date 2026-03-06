import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(batch_size = 64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,))
    ])

    train_dataset = datasets.FashionMNIST(
        root = "data",
        train = True,
        transform = train_transform,
        download = True
    )

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,))
    ])

    test_dataset = datasets.FashionMNIST(
        root = "data",
        train = False,
        transform = test_transform,
        download = True
    )

    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataloader, test_dataloader