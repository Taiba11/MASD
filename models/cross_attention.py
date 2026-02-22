"""
Cross-Attention Adaptive Fusion for MASD (Section II-B.3, Eq. 8).

Fuses learned SSL features (f_SSL in R^130) with handcrafted features (f_hand in R^3)
using scaled dot-product cross-attention:

    Q = W_Q * f_SSL,  K = W_K * f_hand,  V = W_V * f_hand
    alpha = softmax(Q @ K^T / sqrt(d_k))
    h_att = alpha @ V

    f_fused = W_o * [f_SSL ; h_att]  in R^256

Automatically emphasizes phase coherence for classical vocoders
and spectral features for neural synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of SSL and handcrafted features.

    Args:
        ssl_dim (int): Dimension of SSL features (default: 130).
        hand_dim (int): Dimension of handcrafted features (default: 3).
        dk (int): Key/query dimension (default: 64).
        dv (int): Value dimension (default: 64).
        output_dim (int): Final fused output dimension (default: 256).
    """

    def __init__(
        self,
        ssl_dim: int = 130,
        hand_dim: int = 3,
        dk: int = 64,
        dv: int = 64,
        output_dim: int = 256,
    ):
        super().__init__()
        self.dk = dk

        # Projection matrices
        self.W_Q = nn.Linear(ssl_dim, dk)      # Query from SSL features
        self.W_K = nn.Linear(hand_dim, dk)      # Key from handcrafted features
        self.W_V = nn.Linear(hand_dim, dv)      # Value from handcrafted features
        self.W_o = nn.Linear(ssl_dim + dv, output_dim)  # Output projection

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, f_ssl, f_hand):
        """
        Args:
            f_ssl: (batch, ssl_dim) — SSL features [mu; sigma_tr; kappa].
            f_hand: (batch, hand_dim) — handcrafted features [PCS; HFER; SFI].
        Returns:
            f_fused: (batch, output_dim) — fused representation.
        """
        # Compute Q, K, V
        Q = self.W_Q(f_ssl).unsqueeze(1)    # (batch, 1, dk)
        K = self.W_K(f_hand).unsqueeze(1)   # (batch, 1, dk)
        V = self.W_V(f_hand).unsqueeze(1)   # (batch, 1, dv)

        # Scaled dot-product attention (Eq. 8)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.dk ** 0.5)
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        h_att = torch.bmm(alpha, V).squeeze(1)  # (batch, dv)

        # Concatenate and project
        concat = torch.cat([f_ssl, h_att], dim=-1)  # (batch, ssl_dim + dv)
        f_fused = self.W_o(concat)  # (batch, output_dim)
        f_fused = self.layer_norm(f_fused)

        return f_fused

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
