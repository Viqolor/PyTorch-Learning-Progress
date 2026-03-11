import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, dataloader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, costs, labels in dataloader:
            images, costs = images.to(device), costs.to(device)
            outputs = model(images, costs)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize = (12, 10))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = classes, yticklabels = classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Multimodal ResNet: Confusion Matrix')
    plt.savefig('Confusion Matrix.png', dpi = 300, bbox_inches = 'tight')
    print(f"Confusion Matrix saved to 'Confusion Matrix.png")

def toggle_backbone(model, freeze = True):
    for param in model.stem.parameters():
        param.requires_grad = not freeze
    for param in model.layers.parameters():
        param.requires_grad = not freeze
        
    status = "FROZEN (Training Head Only)" if freeze else "UNFROZEN (Fine-Tuning)"
    print(f"Model Backbone is now {status}")

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameter count: {total:,} | Trainable parameter count: {trainable:,}")