import sys

import torch
from torch import optim, cuda, OutOfMemoryError
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork
from data_loader import get_dataloaders
from loss import get_loss
from train_utils import train_epoch, validate

def main():
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print(f"Using {device} device")

    try:
        model = NeuralNetwork().to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"CNN initialized \nTotal Parameters: {param_count}")
    except OutOfMemoryError as e:
        print(f"GPU Memory is full: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

    criterion = get_loss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    try:
        train_dataloader, test_dataloader = get_dataloaders(batch_size = 64)
        print(f"Training data and testing data has been loaded successfully")
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

    best = 0.0

    for epoch_count in range(1, 11):
        train_epoch(model, train_dataloader, device, criterion, optimizer, epoch_count)
        accuracy = validate(model, test_dataloader, device, criterion)

        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), "best_cnn_model.pth")
            print(f"New best validation accuracy ({best:.1f}%) \nModel saved")

    print(f"Training complete \nBest validation accuracy: {best:.1f}%")

if __name__ == "__main__":
    main()