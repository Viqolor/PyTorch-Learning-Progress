import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.make_layer(32, 2, stride = 1)
        self.layer2 = self.make_layer(64, 2, stride = 2)
        self.layer3 = self.make_layer(128, 2, stride = 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        def print_shape(name, tensor):
            if not hasattr(self, "_printed"):
                print(f"{name}: {tensor.shape}")

        out = F.relu(self.bn1(self.conv1(x)))
        print_shape("Stem", out)

        out = self.layer1(out)
        print_shape("Layer 1", out)

        out = self.layer2(out)
        print_shape("Layer 2", out)

        out = self.layer3(out)
        print_shape("Layer 3", out)

        out = self.avg_pool(out)
        print_shape("Global_Pool", out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        print_shape("FC", out)

        self._printed = True

        return out