import sys
import pandas as pd

import torch
from torch import optim, cuda, OutOfMemoryError

from model import NeuralNetwork
from data_loader import get_dataloader
from loss import get_loss
from train import train_epoch, validate_epoch

def main():
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print(f"Using {device} device")

    try:
        Simple_CNN = NeuralNetwork().to(device)
        param_count = sum(p.numel() for p in Simple_CNN.parameters())
        print(f"CNN initialized \nTotal parameters: {param_count}")
    except OutOfMemoryError as e:
        print(f"GPU memory is full: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

    criterion = get_loss(device)
    optimizer = optim.Adam(Simple_CNN.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 2)
    history = {
        "train loss": [],
        "train accuracy": [],
        "validation loss": [],
        "validation accuracy": []
    }

    try:
        train_dataloader, test_dataloader = get_dataloader(batch_size = 64)
        print(f"Training data and testing data has been loaded successfully")
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

    best = 0.0

    for epoch_count in range(1,11):
        train_loss, train_accuracy = train_epoch(Simple_CNN, train_dataloader, device, criterion, optimizer, epoch_count)
        val_loss, val_accuracy = validate_epoch(Simple_CNN, test_dataloader, device, criterion)
        scheduler.step(val_loss)

        history["train loss"].append(train_loss)
        history["train accuracy"].append(train_accuracy)
        history["validation loss"].append(val_loss)
        history["validation accuracy"].append(val_accuracy)

        if val_accuracy > best:
            best = val_accuracy
            torch.save(Simple_CNN.state_dict(), "test_best_cnn_model.pth")
            print(f"New best validation accuracy ({best:.1f}%) \nModel saved")

    print(f"Training complete \nBest validation accuracy: {val_accuracy:.1f}%")

    df = pd.DataFrame(history)
    df.index.name = "epoch"
    df.index += 1
    df.to_csv("test_model_history.csv")
    print(f"Model history saved to 'test_model_history.csv'")

if __name__=="__main__":
    main()