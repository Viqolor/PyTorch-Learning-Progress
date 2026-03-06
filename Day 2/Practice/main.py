import sys
import pandas as pd

import torch
from torch import optim, cuda, OutOfMemoryError

from model import NeuralNetwork
from data_loader import get_dataloaders
from loss import get_loss
from train_utils import train_epoch, validate_epoch

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

    criterion = get_loss(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 2)
    history = {
        "train loss":[], 
        "train accuracy":[],
        "validation loss":[], 
        "validation accuracy":[]
    }

    try:
        train_dataloader, test_dataloader = get_dataloaders(batch_size = 64)
        print(f"Training data and testing data has been loaded successfully")
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

    best = 0.0

    for epoch_count in range(1, 11):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, device, criterion, optimizer, epoch_count)
        val_loss, val_accuracy = validate_epoch(model, test_dataloader, device, criterion)
        scheduler.step(val_loss)

        history["train loss"].append(train_loss)
        history["train accuracy"].append(train_accuracy)
        history["validation loss"].append(val_loss)
        history["validation accuracy"].append(val_accuracy)

        if val_accuracy > best:
            best = val_accuracy
            torch.save(model.state_dict(), "best_cnn_model.pth")
            print(f"New best validation accuracy ({best:.1f}%) \nModel saved")

    print(f"Training complete \nBest validation accuracy: {best:.1f}%")

    df = pd.DataFrame(history)
    df.index.name = 'Epoch'
    df.index += 1
    df.to_csv("model_history.csv")
    print(f"Model history saved to 'model_history.csv'")
    
if __name__ == "__main__":
    main()