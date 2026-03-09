import torch
from torch import nn

def get_loss(device, weights = None):
    if weights is not None:
        weight_tensor = torch.tensor(weights).to(device)
        return nn.CrossEntropyLoss(weights = weight_tensor)
    return nn.CrossEntropyLoss()