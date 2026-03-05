import math
import sys
import pandas as pd
import numpy as np

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_loss():
    return nn.CrossEntropyLoss()