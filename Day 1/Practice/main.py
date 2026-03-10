import math
import sys
import pandas as pd
import numpy as np

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork
from data_loader import get_dataloaders
from loss import get_loss

def train_epoch(model, dataloader, device, criterion, optimizer, epoch_count):
    model.train()
    print(f"Epoch #{epoch_count}")
    num_batches = len(dataloader)
    train_loss = 0
    batch_loss = 0

    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_loss += loss.item()
        
        if batch % 100 == 99:
            print(f"Batch {batch+1}, Batch Loss: {batch_loss / 100:.8f}")
            batch_loss = 0.0
    
    train_loss /= num_batches
    print(f"Training Result \n Current LR: {optimizer.param_groups[0]['lr']}, Train Loss {train_loss:.8f}")

def validate(model, dataloader, device, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Validation Result \n Accuracy: {(100*correct):.1f}%, Val Loss {val_loss:.8f}")

def main():
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    Practice_MLP = NeuralNetwork().to(device)
    param_count = sum(p.numel() for p in Practice_MLP.parameters())
    print(f"Total Parameters: {param_count}")
    train_loader, test_loader = get_dataloaders(batch_size = 64)
    print(f"Training data has been loaded successfully")
    
    criterion = get_loss()
    optimizer = optim.Adam(Practice_MLP.parameters(), lr = 1e-3)

    for epoch_count in range(1, 11):
        train_epoch(Practice_MLP, train_loader, device, criterion, optimizer, epoch_count)
        validate(Practice_MLP, test_loader, device, criterion)

if __name__ == "__main__":
    main()