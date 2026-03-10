import torch
from torch import nn

def get_loss(device, weights = None):
    if weights is not None:
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight = weights_tensor, label_smoothing = 0.1)
    return nn.CrossEntropyLoss(label_smoothing = 0.1)