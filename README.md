<div align="center">

# MASD: Multi-Scale Artifact-Aware Self-Supervised Deepfake Detector

[![Paper](https://img.shields.io/badge/Paper-IEEE%20SPL%202026-blue.svg)](https://doi.org/10.1109/LSP.2025.3634032)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Journal](https://img.shields.io/badge/IEEE%20Signal%20Processing%20Letters-Vol.%2033-purple.svg)](#)

**Official implementation of the paper published in IEEE Signal Processing Letters, Vol. 33, 2026**

[Taiba Majid Wani](mailto:majid@diag.uniroma1.it), Member, IEEE &nbsp;&nbsp;&nbsp;
[Irene Amerini](mailto:amerini@diag.uniroma1.it), Member, IEEE

Sapienza University of Rome, Italy

<br>

<img src="assets/architecture.png" alt="MASD Architecture" width="850"/>

</div>

---

## Abstract

Neural vocoders enable highly realistic synthetic speech that challenges multimedia authentication. We propose **MASD**, combining multi-scale SSL with handcrafted features. MASD decomposes spectrograms into **three frequency bands**, processed through an encoder pretrained using **masked reconstruction**, **contrastive predictive coding**, and **adversarial vocoder classification**. Features fuse with phase coherence, spectral flux, and high-frequency energy through **cross-attention**, classified by **temperature-scaled SVM**. Achieves **0.39% EER** and **98.23% accuracy** on ASVspoof 2019 LA.

### Key Results

| Dataset | EER (%) | Accuracy | F1-Score |
|---------|---------|----------|----------|
| **ASVspoof 2019 LA** | **0.39** | **98.23%** | **97.84%** |
| FoR (zero-shot) | 5.93 | - | - |
| ASVspoof 2021 DF (zero-shot) | 7.91 | - | - |
| WaveFake (zero-shot) | 6.42 | - | - |

---

## Highlights

- **Multi-Scale Spectral Decomposition** - 3-band frequency decomposition (0-1 kHz, 1-4 kHz, 4-8 kHz)
- **Adversarial Vocoder Augmentation** - Progressive augmentation across 6 vocoders with gradient reversal
- **Cross-Attention Feature Fusion** - SSL embeddings + handcrafted features (PCS, HFER, SFI)
- **Temperature-Calibrated SVM** - 1.1% ECE for deployment-ready confidence
- **Zero-Shot Generalization** - Validated on FoR, ASVspoof 2021 DF, WaveFake

---

## Architecture

```
Raw Audio x(t)
    |
    +----------------------------------------------+
    |                                              |
    v                                              v
 Log-Mel Spectrogram                     Handcrafted Features
 F=64, 25ms window                       PCS + HFER + SFI
    |
    +--------+--------+
    v        v        v
  S_low    S_mid    S_high     3-Band Decomposition
  0-1kHz   1-4kHz   4-8kHz
    |        |        |
    v        v        v
  Shared CNN Encoder E(.)       5 ResBlocks, 64->128->256->256->128
  (frozen at inference)
    |
    v
  z_fused (Eq.1)  -->  f_SSL = [mu; sigma; kappa] in R^130
    |                          |
    |          +---------------+
    v          v
  Cross-Attention Fusion (Eq.8)    Q=SSL, K/V=Handcrafted
    |
    v  f_fused in R^256
  Temperature-Scaled SVM (Eq.9-10)   RBF kernel, C=10
    |
    v
  Real/Fake + Confidence (ECE=1.1%)
```

---

## Installation

```bash
git clone https://github.com/TaibaMajidWani/MASD.git
cd MASD
conda create -n masd python=3.9 -y && conda activate masd
pip install -r requirements.txt
pip install -e .
```

---

## Training

### Stage 1: SSL Pretraining

```bash
python pretrain/ssl_pretrain.py \
    --config configs/pretrain.yaml \
    --data_dir data/LibriSpeech/train-clean-360 \
    --output_dir checkpoints/ssl_pretrained \
    --epochs 100 --batch_size 256 --lr 1e-4
```

### Stage 2: Downstream Training

```bash
python scripts/train.py \
    --config configs/downstream.yaml \
    --encoder_ckpt checkpoints/ssl_pretrained/best_encoder.pth \
    --data_dir data/ASVspoof2019/LA \
    --output_dir experiments/masd
```

---

## Evaluation

```bash
# In-domain
python scripts/evaluate.py \
    --checkpoint experiments/masd/ \
    --data_dir data/ASVspoof2019/LA --split eval

# Cross-dataset zero-shot
python scripts/evaluate.py \
    --checkpoint experiments/masd/ \
    --data_dir data/WaveFake --dataset wavefake --zero_shot
```

### Single File Inference

```bash
python scripts/inference.py \
    --checkpoint experiments/masd/ \
    --audio_path path/to/audio.wav
```

---

## Results

### ASVspoof 2019 LA (Table I)

| Method | Accuracy | EER (%) | F1-Score |
|--------|----------|---------|----------|
| RawNet2 | 95.03% | 2.11 | 94.21% |
| AASIST | 96.8% | 1.21 | 96.0% |
| **MASD (Ours)** | **98.23%** | **0.39** | **97.84%** |

### Ablation Study (Table II)

| Configuration | EER (%) | Delta (%) |
|--------------|---------|-----------|
| Handcrafted only | 5.63 | +1344 |
| SSL (single-scale) | 2.14 | +449 |
| SSL (multi-scale) | 1.37 | +251 |
| + Handcrafted (attention) | 1.52 | +290 |
| **Full MASD** | **0.39** | **-** |
| w/o adversarial | 1.52 | +290 |

### State-of-the-Art Comparison (Table III)

| Method | Framework | EER (%) |
|--------|-----------|---------|
| SAMO | Supervised | 1.09 |
| Wav2vec + AASIST2 | SSL | 1.61 |
| HuBert + WavLM | SSL | 0.42 |
| **MASD (Ours)** | **SSL** | **0.39** |

---

## Citation

```bibtex
@article{wani2026masd,
    title     = {Multi-Scale Self-Supervised Learning for Efficient Audio Deepfake Detection},
    author    = {Wani, Taiba Majid and Amerini, Irene},
    journal   = {IEEE Signal Processing Letters},
    volume    = {33},
    pages     = {46--50},
    year      = {2026},
    doi       = {10.1109/LSP.2025.3634032},
    publisher = {IEEE}
}
```

## Acknowledgments

Supported by **SERICS** (PE00000014) through MUR National Recovery and Resilience Plan funded by EU - NextGenerationEU.

## License

MIT License - see [LICENSE](LICENSE) for details.
