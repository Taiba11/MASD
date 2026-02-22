"""
MASD: Multi-Scale Artifact-Aware Self-Supervised Deepfake Detector.

Full pipeline (Section II):
    Stage 1: SSL Pretraining — multi-scale encoder with masked recon + CPC + adversarial
    Stage 2: Feature Extraction — f_SSL (mean, std, skew) + f_hand (PCS, HFER, SFI)
    Stage 3: Cross-Attention Fusion + Temperature-Calibrated SVM

Paper: Wani & Amerini, IEEE Signal Processing Letters, Vol. 33, 2026
"""

import numpy as np
import torch
import torch.nn as nn

from .spectral_decomposition import SpectralDecomposer
from .encoder import SharedCNNEncoder
from .cross_attention import CrossAttentionFusion
from .handcrafted_features import HandcraftedFeatureExtractor


class MASD(nn.Module):
    """
    Full MASD framework (neural components).

    For inference:
        1. Compute log-Mel spectrogram
        2. Decompose into 3 bands
        3. Encode with frozen shared encoder
        4. Compute f_SSL = [mu, sigma_tr, kappa]
        5. Compute f_hand = [PCS, HFER, SFI]
        6. Cross-attention fusion -> f_fused (256-dim)
        7. Pass to calibrated SVM (external)

    Args:
        sample_rate (int): Audio sample rate.
        n_mels (int): Mel bins.
        embedding_dim (int): Encoder embedding dim d.
        output_dim (int): Fused output dimension.
        freeze_encoder (bool): Freeze encoder weights at downstream.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        embedding_dim: int = 128,
        output_dim: int = 256,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Spectral decomposition
        self.decomposer = SpectralDecomposer(
            sample_rate=sample_rate, n_mels=n_mels,
        )

        # Shared CNN encoder (frozen at inference)
        self.encoder = SharedCNNEncoder(
            in_channels=1, embedding_dim=embedding_dim,
        )

        # Learnable band importance weights (Eq. 1)
        self.band_weights = nn.Parameter(torch.zeros(3))

        # Cross-attention fusion
        ssl_dim = embedding_dim + 2  # mu(128) + sigma(1) + kappa(1) = 130
        self.attention = CrossAttentionFusion(
            ssl_dim=ssl_dim, hand_dim=3, output_dim=output_dim,
        )

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode_bands(self, spectrogram):
        """
        Encode 3-band spectrogram through shared encoder with learnable fusion (Eq. 1).

        Args:
            spectrogram: (B, 1, F, T) — log-Mel spectrogram.
        Returns:
            z_fused: (B, T', d) — fused temporal embeddings.
        """
        bands = self.decomposer(spectrogram)
        alpha = torch.softmax(self.band_weights, dim=0)

        embeddings = []
        for i, name in enumerate(["low", "mid", "high"]):
            z = self.encoder(bands[name])  # (B, T', d)
            embeddings.append(alpha[i] * z)

        # z_fused = sum_j alpha^(j) * z_bar^(j) pooled over time
        z_fused = sum(embeddings)  # (B, T', d)
        return z_fused

    def compute_ssl_features(self, z):
        """
        Compute statistical SSL features: f_SSL = [mu; sigma_tr; kappa] (Section II-B.1).

        Args:
            z: (B, T', d) — temporal embeddings.
        Returns:
            f_ssl: (B, d+2) = (B, 130) — SSL feature vector.
        """
        mu = z.mean(dim=1)  # (B, d)
        diff = z - mu.unsqueeze(1)
        norms = torch.norm(diff, dim=-1)  # (B, T')

        sigma_tr = torch.sqrt((norms ** 2).mean(dim=1, keepdim=True) + 1e-8)  # (B, 1)
        kappa = ((norms / (sigma_tr.squeeze(1) + 1e-8)) ** 3).mean(dim=1, keepdim=True)  # (B, 1)

        f_ssl = torch.cat([mu, sigma_tr, kappa], dim=-1)  # (B, d+2)
        return f_ssl

    def forward(self, spectrogram, f_hand):
        """
        Full forward pass (neural components).

        Args:
            spectrogram: (B, 1, F, T) — log-Mel spectrogram.
            f_hand: (B, 3) — handcrafted features tensor.
        Returns:
            f_fused: (B, 256) — fused features for SVM.
        """
        z = self.encode_bands(spectrogram)
        f_ssl = self.compute_ssl_features(z)
        f_fused = self.attention(f_ssl, f_hand)
        return f_fused

    def extract_features(self, spectrogram, f_hand):
        """Extract fused features as numpy for SVM (inference mode)."""
        with torch.no_grad():
            f_fused = self.forward(spectrogram, f_hand)
        return f_fused.cpu().numpy()
