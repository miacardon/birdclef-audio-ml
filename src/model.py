import torch
import torch.nn as nn
import torch.nn.functional as F


# define model class
class BirdClefCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CONVOLUTIONAL FEATURE EXTRACTOR
        # detect simple edges/harmonics
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # combines simple features into patterns
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # high-level bird call signatures
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        # global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classification head
        self.classifier = nn.Linear(128, num_classes)

    # define forward pass
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x
