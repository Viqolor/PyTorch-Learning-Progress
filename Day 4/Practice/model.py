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

        self.tab_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True)
        )

        self.fc = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Linear(128 + 32, num_classes)
            )

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, images, costs):
        def print_shape(name, tensor):
            if not hasattr(self, "_printed"):
                print(f"{name}: {tensor.shape}")

        img_features = F.relu(self.bn1(self.conv1(images)))
        print_shape(f"Stem", img_features)

        img_features = self.layer1(img_features)
        print_shape(f"Layer 1", img_features)

        img_features = self.layer2(img_features)
        print_shape(f"Layer 2", img_features)

        img_features = self.layer3(img_features)
        print_shape(f"Layer 3", img_features)

        img_features = self.avg_pool(img_features)
        print_shape(f"Global_Pool", img_features)

        img_features = torch.flatten(img_features, 1)
        tab_features = self.tab_fc(costs)

        combined = torch.cat((img_features, tab_features), dim = 1)
        print_shape(f"Concat", combined)

        out = self.fc(combined)
        print_shape(f"FC", out)

        self._printed = True

        return out