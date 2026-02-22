"""
Spectral Multi-Scale Decomposition (Section II-A.1).

Decomposes log-Mel spectrogram S into three sub-bands:
    S_low  (0-1 kHz):  fundamental frequency artifacts
    S_mid  (1-4 kHz):  formant irregularities
    S_high (4-8 kHz):  vocoder-specific distortions (aliasing, discontinuities)
"""

import torch
import torch.nn as nn
import numpy as np


class SpectralDecomposer(nn.Module):
    """
    Decomposes a log-Mel spectrogram into three frequency sub-bands.

    Args:
        sample_rate (int): Audio sample rate.
        n_mels (int): Number of Mel bins (F).
        bands (dict): Frequency ranges for each band in Hz.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        bands: dict = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.bands = bands or {
            "low": (0, 1000),
            "mid": (1000, 4000),
            "high": (4000, 8000),
        }

        # Precompute mel bin indices for each band
        mel_freqs = self._mel_frequencies(n_mels, 0, sample_rate // 2)
        self.band_indices = {}
        for name, (f_lo, f_hi) in self.bands.items():
            mask = (mel_freqs >= f_lo) & (mel_freqs < f_hi)
            indices = torch.where(torch.tensor(mask))[0]
            if len(indices) == 0:
                # Fallback: at least 1 bin
                closest = torch.argmin(torch.abs(torch.tensor(mel_freqs) - (f_lo + f_hi) / 2))
                indices = torch.tensor([closest])
            self.register_buffer(f"idx_{name}", indices)

    @staticmethod
    def _mel_frequencies(n_mels, fmin, fmax):
        """Compute center frequencies of Mel filter banks."""
        mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
        mels = np.linspace(mel_min, mel_max, n_mels)
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def forward(self, S):
        """
        Decompose spectrogram into sub-bands.

        Args:
            S: (batch, 1, F, T) — log-Mel spectrogram.
        Returns:
            dict with 'low', 'mid', 'high' tensors, each (batch, 1, F_band, T).
        """
        result = {}
        for name in self.bands:
            idx = getattr(self, f"idx_{name}")
            result[name] = S[:, :, idx, :]
        return result
