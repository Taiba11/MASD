"""
Handcrafted Feature Extraction for MASD (Section II-B.2).

Three interpretable acoustic features targeting known synthesis artifacts:

1. Phase Coherence Score (PCS) - Eq. 6:
   phi_PCS = (1/TF) * sum_t sum_w |unwrapped_phase(t, w+1) - unwrapped_phase(t, w)|
   Vocoders introduce phase jumps due to independent frame synthesis.

2. High-Frequency Energy Ratio (HFER) - Eq. 7:
   phi_HFER = sum_{f>8kHz} |X(f)|^2 / sum_f |X(f)|^2
   Neural vocoders often amplify or suppress high frequencies unnaturally.

3. Spectral Flux Irregularity (SFI):
   SF_t = sum_f max(0, |S_t(f)| - |S_{t-1}(f)|)
   phi_SFI = sqrt(Var({SF_t})) / Mean({SF_t})
   Synthetic speech exhibits unnatural spectral flux patterns.

Output: f_hand = [phi_PCS; phi_HFER; phi_SFI] in R^3
"""

import numpy as np
import torch
import librosa


class HandcraftedFeatureExtractor:
    """
    Extracts three handcrafted acoustic features from raw audio.

    Args:
        sample_rate (int): Audio sample rate.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        hf_cutoff (float): High-frequency cutoff in Hz for HFER.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        hf_cutoff: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.hf_cutoff = hf_cutoff

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract all three handcrafted features.

        Args:
            waveform: 1D numpy array of audio samples.
        Returns:
            f_hand: (3,) array [phi_PCS, phi_HFER, phi_SFI].
        """
        stft = librosa.stft(y=waveform, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        pcs = self._phase_coherence_score(phase)
        hfer = self._high_frequency_energy_ratio(magnitude)
        sfi = self._spectral_flux_irregularity(magnitude)

        return np.array([pcs, hfer, sfi], dtype=np.float32)

    def _phase_coherence_score(self, phase: np.ndarray) -> float:
        """Phase Coherence Score (Eq. 6)."""
        # Unwrap phase across frequency axis
        unwrapped = np.unwrap(phase, axis=0)
        # Frequency derivative via forward differences
        freq_deriv = np.abs(np.diff(unwrapped, axis=0))
        T, F = phase.shape[1], phase.shape[0]
        pcs = freq_deriv.sum() / (T * max(F - 1, 1))
        return float(pcs)

    def _high_frequency_energy_ratio(self, magnitude: np.ndarray) -> float:
        """High-Frequency Energy Ratio (Eq. 7)."""
        freqs = np.linspace(0, self.sample_rate / 2, magnitude.shape[0])
        hf_mask = freqs > self.hf_cutoff
        energy_total = (magnitude ** 2).sum()
        energy_hf = (magnitude[hf_mask] ** 2).sum() if hf_mask.any() else 0.0
        return float(energy_hf / max(energy_total, 1e-10))

    def _spectral_flux_irregularity(self, magnitude: np.ndarray) -> float:
        """Spectral Flux Irregularity (SFI)."""
        # Frame-wise spectral flux: SF_t = sum_f max(0, |S_t| - |S_{t-1}|)
        diff = np.maximum(0, magnitude[:, 1:] - magnitude[:, :-1])
        sf = diff.sum(axis=0)  # (T-1,)
        mean_sf = sf.mean()
        if mean_sf < 1e-10:
            return 0.0
        return float(np.sqrt(sf.var()) / mean_sf)

    def extract_batch(self, waveforms: list) -> np.ndarray:
        """Extract features for a batch of waveforms."""
        return np.stack([self.extract(w) for w in waveforms])
