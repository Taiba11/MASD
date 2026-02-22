"""
Multi-Task Self-Supervised Losses for MASD (Section II-A.3, Eq. 3-5).

Three complementary SSL tasks:
    1. Masked Spectrogram Reconstruction (L_mask) - L2 loss on 15% masked patches
    2. Contrastive Predictive Coding (L_CPC) - Eq. 4, InfoNCE loss
    3. Adversarial Vocoder Classification (L_adv) - Eq. 3, binary + GRL(vocoder)

Joint optimization: L_total = L_mask + L_CPC + L_adv    (Eq. 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedReconstructionLoss(nn.Module):
    """
    Masked spectrogram reconstruction loss.

    Randomly masks 15% of 16x16 time-frequency patches and
    reconstructs them using a lightweight decoder with L2 loss.
    """

    def __init__(self, mask_ratio=0.15, patch_size=16):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def create_mask(self, S):
        """Create random patch mask for spectrogram."""
        B, C, F, T = S.shape
        pf = max(1, F // self.patch_size)
        pt = max(1, T // self.patch_size)
        num_patches = pf * pt
        num_masked = max(1, int(num_patches * self.mask_ratio))

        mask = torch.ones(B, 1, F, T, device=S.device)
        for b in range(B):
            indices = torch.randperm(num_patches)[:num_masked]
            for idx in indices:
                pi, pj = idx // pt, idx % pt
                fi = pi * self.patch_size
                ti = pj * self.patch_size
                fe = min(fi + self.patch_size, F)
                te = min(ti + self.patch_size, T)
                mask[b, 0, fi:fe, ti:te] = 0.0
        return mask

    def forward(self, S_original, S_reconstructed, mask):
        """
        Args:
            S_original: (B, 1, F, T) — original spectrogram.
            S_reconstructed: (B, 1, F, T) — decoder output.
            mask: (B, 1, F, T) — 0 at masked locations.
        Returns:
            L2 loss on masked regions only.
        """
        inv_mask = 1.0 - mask
        masked_pixels = inv_mask.sum().clamp(min=1.0)
        loss = ((S_original - S_reconstructed) ** 2 * inv_mask).sum() / masked_pixels
        return loss


class CPCLoss(nn.Module):
    """
    Contrastive Predictive Coding loss (Eq. 4).

    L_CPC = -sum_t log( exp(sim(c_t, z_{t+k}) / tau) /
                         sum_{j in N} exp(sim(c_t, z_j) / tau) )

    Args:
        prediction_steps (int): Number of steps to predict ahead (k=12).
        n_negatives (int): Number of negative samples (128).
        temperature (float): InfoNCE temperature (tau=0.07).
    """

    def __init__(self, prediction_steps=12, n_negatives=128, temperature=0.07):
        super().__init__()
        self.k = prediction_steps
        self.n_negatives = n_negatives
        self.tau = temperature

    def forward(self, context, embeddings):
        """
        Args:
            context: (B, T, d) — GRU context representations c_t.
            embeddings: (B, T, d) — encoder embeddings z_t.
        Returns:
            CPC loss scalar.
        """
        B, T, d = embeddings.shape
        if T <= self.k:
            return torch.tensor(0.0, device=embeddings.device)

        loss = 0.0
        count = 0

        for t in range(T - self.k):
            c_t = context[:, t, :]  # (B, d)
            z_pos = embeddings[:, t + self.k, :]  # (B, d) — positive

            # Negative sampling from batch
            neg_indices = torch.randint(0, B * T, (B, self.n_negatives), device=embeddings.device)
            flat_emb = embeddings.reshape(-1, d)
            neg_indices = neg_indices.clamp(max=flat_emb.size(0) - 1)
            z_neg = flat_emb[neg_indices.reshape(-1)].reshape(B, self.n_negatives, d)

            # Cosine similarity
            pos_sim = F.cosine_similarity(c_t, z_pos, dim=-1) / self.tau  # (B,)
            neg_sim = F.cosine_similarity(
                c_t.unsqueeze(1), z_neg, dim=-1
            ) / self.tau  # (B, n_neg)

            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+n_neg)
            labels = torch.zeros(B, dtype=torch.long, device=embeddings.device)
            loss += F.cross_entropy(logits, labels)
            count += 1

        return loss / max(count, 1)


class AdversarialLoss(nn.Module):
    """
    Adversarial vocoder classification loss (Eq. 3).

    L_adv = L_binary + lambda_adv * GRL(L_vocoder)

    Binary loss distinguishes real vs. vocoded.
    Vocoder loss classifies which vocoder (6-way).
    GRL ensures encoder learns vocoder-invariant features.

    Args:
        num_vocoders (int): Number of vocoder types (default: 6).
        lambda_adv (float): GRL scaling factor (default: 0.3).
    """

    def __init__(self, num_vocoders=6, lambda_adv=0.3):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.vocoder_loss = nn.CrossEntropyLoss()

    def forward(self, binary_logits, binary_labels, vocoder_logits=None, vocoder_labels=None):
        """
        Args:
            binary_logits: (B,) — real/vocoded predictions.
            binary_labels: (B,) — 0=real, 1=vocoded.
            vocoder_logits: (B, num_vocoders) — vocoder type predictions (post-GRL).
            vocoder_labels: (B,) — vocoder type indices.
        Returns:
            L_adv scalar.
        """
        l_binary = self.binary_loss(binary_logits, binary_labels.float())

        if vocoder_logits is not None and vocoder_labels is not None:
            # Vocoder classification loss (through GRL)
            mask = binary_labels > 0  # Only for vocoded samples
            if mask.sum() > 0:
                l_vocoder = self.vocoder_loss(vocoder_logits[mask], vocoder_labels[mask])
                return l_binary + self.lambda_adv * l_vocoder

        return l_binary
