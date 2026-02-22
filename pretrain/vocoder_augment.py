"""
Progressive Vocoder Augmentation for MASD (Section II-A.2, Eq. 2).

P(theta_k | e) = exp(gamma * e * d(theta_k)) / sum_k' exp(gamma * e * d(theta_k'))

gamma = 0.01, d(theta_k) in {1, 2, 3} indicates vocoder sophistication.
Advanced vocoders sampled 4.35x more frequently by epoch 50 than epoch 1.

Vocoders: WORLD, WaveGlow, MelGAN, HiFi-GAN, UnivNet, BigVGAN
"""

import numpy as np


# Vocoder difficulty levels
VOCODER_CONFIG = {
    "WORLD": {"difficulty": 1, "description": "Classical signal processing"},
    "WaveGlow": {"difficulty": 1, "description": "Flow-based autoregressive"},
    "MelGAN": {"difficulty": 2, "description": "GAN-based non-autoregressive"},
    "HiFi-GAN": {"difficulty": 2, "description": "GAN with multi-period discriminator"},
    "UnivNet": {"difficulty": 3, "description": "Multi-resolution spectrogram discriminator"},
    "BigVGAN": {"difficulty": 3, "description": "Large-scale universal vocoder"},
}

GAMMA = 0.01


def get_vocoder_sampling_probs(epoch: int, gamma: float = GAMMA) -> dict:
    """
    Compute progressive vocoder sampling probabilities (Eq. 2).

    Args:
        epoch: Current training epoch.
        gamma: Scheduling parameter (default: 0.01).
    Returns:
        dict mapping vocoder name to sampling probability.
    """
    names = list(VOCODER_CONFIG.keys())
    difficulties = [VOCODER_CONFIG[n]["difficulty"] for n in names]

    logits = [gamma * epoch * d for d in difficulties]
    max_logit = max(logits)
    exp_logits = [np.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)

    return {n: e / total for n, e in zip(names, exp_logits)}


def sample_vocoder(epoch: int, gamma: float = GAMMA) -> str:
    """Sample a vocoder according to progressive difficulty scheduling."""
    probs = get_vocoder_sampling_probs(epoch, gamma)
    names = list(probs.keys())
    p = list(probs.values())
    return np.random.choice(names, p=p)


def apply_vocoder_augmentation(waveform: np.ndarray, vocoder_name: str,
                                sample_rate: int = 16000) -> np.ndarray:
    """
    Apply vocoder augmentation to a waveform.

    In practice, this requires pre-vocoded data or vocoder inference.
    This function provides the interface and falls back to simple
    simulation for demonstration.
    """
    # Placeholder: in full implementation, load pre-vocoded audio
    # or run vocoder inference. Here we simulate with simple transforms.
    if vocoder_name in ["WORLD", "WaveGlow"]:
        # Simple: add subtle noise to simulate basic vocoder artifacts
        noise = np.random.randn(*waveform.shape) * 0.005
        return waveform + noise
    elif vocoder_name in ["MelGAN", "HiFi-GAN"]:
        # Medium: slight spectral coloring
        noise = np.random.randn(*waveform.shape) * 0.003
        return waveform + noise
    else:
        # Advanced: minimal perturbation (harder to detect)
        noise = np.random.randn(*waveform.shape) * 0.001
        return waveform + noise
