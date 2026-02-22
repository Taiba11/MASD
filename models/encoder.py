"""
Shared CNN Encoder E(.) for MASD (Section II-A.1).

5 residual blocks: channels 64 -> 128 -> 256 -> 256 -> 128
Kernel 3x3, stride 2, BatchNorm, ReLU.
Shared parameters across all bands for unified spectral representations.
Output: z^(j) in R^{T x d}, d=128 per band.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with Conv2d, BatchNorm, ReLU, skip connection."""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SharedCNNEncoder(nn.Module):
    """
    Shared convolutional encoder with 5 residual blocks.

    Processes each frequency band independently with shared weights
    to learn unified spectral representations.

    Args:
        in_channels (int): Input channels (1 for spectrogram).
        channels (list): Channel sizes per block.
        embedding_dim (int): Output embedding dimension d.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list = None,
        embedding_dim: int = 128,
    ):
        super().__init__()
        channels = channels or [64, 128, 256, 256, 128]
        self.embedding_dim = embedding_dim

        blocks = []
        prev_ch = in_channels
        for ch in channels:
            blocks.append(ResidualBlock(prev_ch, ch, stride=2))
            prev_ch = ch
        self.blocks = nn.Sequential(*blocks)

        # Adaptive pooling to get (batch, channels[-1], 1, T')
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, F_band, T) — sub-band spectrogram.
        Returns:
            z: (batch, T', d) — temporal sequence of embeddings.
        """
        h = self.blocks(x)
        # (batch, C, F', T')
        h = self.adaptive_pool(h)
        # (batch, C, 1, T')
        h = h.squeeze(2).permute(0, 2, 1)
        # (batch, T', C)
        z = self.proj(h)
        # (batch, T', d)
        return z
