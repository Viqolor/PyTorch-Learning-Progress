import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork
from loss import get_loss
from data_loader import get_dataloaders

def train_epoch(model, dataloader, device, criterion, optimizer, epoch_count):
    num_batches = len(dataloader)
   
    try:
        model.train()
        print(f"Epoch #{epoch_count}")
        train_loss, batch_loss, correct, samples_processed = 0.0, 0.0, 0.0, 0.0

        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_loss += loss.item()

            batch_accuracy = 100 * (output.argmax(1) == labels).sum().item() / len(images)
            correct += (output.argmax(1) == labels).sum().item()
            samples_processed += len(images)
            cum_accuracy = 100 * correct/samples_processed

            if batch % 100 == 99:
                print(f"Batch {batch+1}: batch_loss {(batch_loss / 100):.8f}, batch_accuracy: {batch_accuracy:.1f}%, cumulative_accuracy: {cum_accuracy:.1f}%")
                batch_loss = 0.0
            
        train_loss /= num_batches
        print(f"Epoch complete: \ntrain_loss: {train_loss:.8f}, current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU Memory is full: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)


def validate(model, dataloader, device, criterion):
    num_batches = len(dataloader)
    val_loss, correct = 0.0, 0.0

    try:
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item()
                correct += (output.argmax(1) == labels).type(torch.float).sum().item()
            
            val_loss /= num_batches
            accuracy = 100 *correct / len(dataloader.dataset)
            print(f"Validation complete: \nval_loss: {val_loss:.8f}, cumulative_accuracy: {accuracy:.1f}%")
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU Memory is full: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    try:
        Simple_MLP = NeuralNetwork().to(device)
        param_count = sum(p.numel() for p in Simple_MLP.parameters())
        print(f"Total Parameters: {param_count}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU Memory is full: {e} \nTerminating the code")
        sys.exit(1)

    try:
        train_loader, test_loader = get_dataloaders(batch_size = 64)
        print(f"Training data and testing data loaded successfully")
    except ImportError as e:
        print(f"Data cannot be loaded: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

    criterion = get_loss()
    optimizer = optim.Adam(Simple_MLP.parameters(), lr = 1e-3)

    for epoch_count in range(1, 11):
        train_epoch(Simple_MLP, train_loader, device, criterion, optimizer, epoch_count)
        validate(Simple_MLP, test_loader, device, criterion)
        
if __name__ == "__main__":
    main()