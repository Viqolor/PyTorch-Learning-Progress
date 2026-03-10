import sys

import torch
from torch import optim, cuda
import pandas as pd

from model import ResNet
from dataloader import get_loaders
from loss import get_loss
from train import train_epoch, validate_epoch
from utils import plot_confusion_matrix

def main():
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    if device.type == "cpu":
        sys.exit(1)
    print(f"Using {torch.cuda.get_device_name(0)} device")

    train_loader, test_loader = get_loaders(batch_size = 128)

    model = ResNet(num_classes = 10).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total number of paramters: {param_count}")

    criterion = get_loss(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 2)

    history = {
        'train loss': [],
        'train accuracy': [],
        'val loss': [],
        'val accuracy': []
    }

    best_acc = 0.0

    for epoch in range(1, 11):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, device, optimizer, epoch)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        history['train loss'].append(train_loss)
        history['train accuracy'].append(train_acc)
        history['val loss'].append(val_loss)
        history['val accuracy'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "Best ResNet.pth")
            print(f"New Best Accuracy ({val_acc:.2f}%) Saved")

    df = pd.DataFrame(history)
    df.index.name = "Epoch"
    df.index += 1
    df.to_csv("ResNet Training Log.csv")
    print(f"Training complete\nResults saved to 'ResNet Training Log.csv'")

    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plot_confusion_matrix(model, test_loader, device, classes)

if __name__ == "__main__":
    main()