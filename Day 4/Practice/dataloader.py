import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import torch
import numpy as np

def generate_cost(label):
    base_prices = {0: 20, 1: 50, 2: 45, 3: 35, 4: 120, 5: 25, 6: 60, 7: 110, 8: 15, 9: 95}
    base = base_prices[label]
    sd = base * 0.25
    price = np.random.normal(base, sd)
    price = max(price, 1.0)
    normalized_cost = torch.tensor([price / 200.0], dtype = torch.float32)
    
    return normalized_cost

class MultimodalFashionMNIST(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        cost = generate_cost(label)
        return img, cost, label

def get_loaders(batch_size = 64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    train_dataset = MultimodalFashionMNIST(
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

    test_dataset = MultimodalFashionMNIST(
        root = 'data',
        train = False,
        transform = test_transform,
        download = True
    )

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader