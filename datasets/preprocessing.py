"""
Audio Preprocessing and Log-Mel Spectrogram Extraction for MASD.

Log-Mel spectrogram: S in R^{F x T}, F=64 Mel bins, 25ms Hamming window, 10ms stride.
"""

import numpy as np
import torch
import torchaudio
import librosa


class AudioPreprocessor:
    """Load and preprocess audio to fixed length."""

    def __init__(self, sample_rate=16000, duration=4.0):
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)

    def load(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).numpy()
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        # Normalize
        mx = np.abs(waveform).max()
        if mx > 0:
            waveform = waveform / mx
        # Fix length
        if len(waveform) > self.target_length:
            waveform = waveform[:self.target_length]
        else:
            waveform = np.pad(waveform, (0, max(0, self.target_length - len(waveform))))
        return waveform


class LogMelExtractor:
    """
    Compute log-Mel spectrogram as described in Section II-A.1.

    25ms Hamming window, 10ms stride, F=64 Mel bins.
    """

    def __init__(self, sample_rate=16000, n_mels=64, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, waveform):
        """
        Args:
            waveform: 1D numpy array.
        Returns:
            (1, F, T) tensor — log-Mel spectrogram.
        """
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, window="hamming",
        )
        log_mel = np.log(mel + 1e-9)
        return torch.from_numpy(log_mel).unsqueeze(0).float()
