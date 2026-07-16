import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """3D Residual block with two convolutions and optional downsampling."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 dropout: float = 0.0,
                 normalization: nn.Module = nn.BatchNorm3d,):

        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = normalization(out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut connection (identity or 1x1 conv if dimensions change)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                normalization(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet3DMRI(nn.Module):
    """3D ResNet-like architecture for small MRI volumes."""
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 base_filters: int = 32,
                 dropout: float = 0.3,
                 normalization: nn.Module = nn.BatchNorm3d,):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            normalization(base_filters),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = ResidualBlock3D(base_filters, base_filters, stride=1, dropout=dropout/2, normalization=normalization)
        self.layer2 = ResidualBlock3D(base_filters, base_filters*2, stride=2, dropout=dropout/2, normalization=normalization)
        self.layer3 = ResidualBlock3D(base_filters*2, base_filters*4, stride=2, dropout=dropout, normalization=normalization)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_filters*4, base_filters*2),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_filters*2, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.fc(x)
        return x
