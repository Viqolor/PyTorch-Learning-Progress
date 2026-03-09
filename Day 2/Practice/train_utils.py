import sys

import torch
from torch import OutOfMemoryError

def train_epoch(model, dataloader, device, criterion, optimizer, epoch_count):
    num_batches = len(dataloader)
    model.train()
    print(f"Epoch #{epoch_count}")
    train_loss, batch_loss, correct, samples_processed = 0.0, 0.0, 0, 0
    
    try:
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_loss += loss.item()
            
            pred = (output.argmax(1) == labels).sum().item()
            batch_accuracy = 100 * pred / len(images)
            correct += pred
            samples_processed += len(images)
            cumul_accuracy = 100 * correct / samples_processed

            if batch % 100 == 99:
                print(f"Batch #{batch + 1}: average batch loss: {batch_loss/100:.8f}, batch accuracy: {batch_accuracy:.1f}%, cumulative accuracy: {cumul_accuracy:.1f}%")
                batch_loss = 0.0

        print(f"Epoch complete \nTrain loss: {train_loss/num_batches:.8f}, overall accuracy: {100 * correct / len(dataloader.dataset):.1f}%")
        return train_loss/num_batches, 100 * correct / len(dataloader.dataset)
    except OutOfMemoryError as e:
        print(f"GPU Memory is full: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)

def validate_epoch(model, dataloader, device, criterion):
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0.0, 0

    try:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output, labels).item()
                
                pred = (output.argmax(1) == labels).sum().item()
                correct += pred
                
            print(f"Validation complete \nValidation loss: {val_loss/num_batches:.8f}, overall accuracy: {100 * correct / len(dataloader.dataset):.1f}%")

        return val_loss/num_batches, 100 * correct / len(dataloader.dataset)
    except OutOfMemoryError as e:
        print(f"GPU Memory is full: {e} \nTerminating the code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error occurred: {e} \nTerminating the code")
        sys.exit(1)