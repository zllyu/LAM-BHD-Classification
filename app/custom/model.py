import torch
from torch import nn


class Net(nn.Module):
    """Small 3D CNN for LAM/BHD/Control volume classification."""

    def __init__(self, num_classes: int = 3, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
