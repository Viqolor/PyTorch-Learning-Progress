import torch

def train_epoch(model, dataloader, criterion, device, optimizer, epoch):
    num_batches = len(dataloader)
    model.train()
    cumul_loss, batch_loss, correct, pred, cumul_total, batch_total= 0.0, 0.0, 0, 0, 0, 0
    print(f"Epoch #{epoch}")

    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        cumul_loss += loss.item()
        batch_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        pred += (output.argmax(1) == labels).sum().item()
        cumul_total += len(images)
        batch_total += len(images)

        if batch % 100 == 99:
            print(f"Batch {batch + 1}: batch loss {batch_loss / (batch % 100 + 1):.8f} | batch accuracy {100 * pred / batch_total:.2f}% | cumulative accuracy {100 * correct / cumul_total:.2f}%")
            batch_loss, pred, batch_total = 0.0, 0, 0
    
    correct = 100 * correct / cumul_total
    cumul_loss /= num_batches
    print(f"Training complete: train loss {cumul_loss:.8f} | overall accuracy {correct:.2f}%")
    return cumul_loss, correct

def validate_epoch(model, dataloader, criterion, device):
    num_batches = len(dataloader)
    model.eval()
    loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss += criterion(output, labels).item()
            correct += (output.argmax(1) == labels).sum().item()
            total += len(images)

    correct = 100 * correct / total
    loss /= num_batches
    print(f"Validation complete: validation loss {loss:.8f} | overall accuracy {correct:.2f}%")
    return loss, correct
            