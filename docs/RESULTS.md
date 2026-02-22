# Detailed Experimental Results

## ASVspoof 2019 LA Performance (Table I)

| Method | Accuracy (%) | EER (%) | F1-Score (%) |
|--------|-------------|---------|-------------|
| RawNet2 | 95.03 | 2.11 | 94.21 |
| AASIST | 96.8 | 1.21 | 96.0 |
| **MASD (Ours)** | **98.23** | **0.39** | **97.84** |

---

## Ablation Study (Table II)

| Configuration | EER (%) | Delta (%) |
|--------------|---------|-----------|
| Handcrafted only | 5.63 | +1344 |
| SSL (single-scale) | 2.14 | +449 |
| SSL (multi-scale) | 1.37 | +251 |
| + Handcrafted (concat) | 1.67 | +328 |
| + Handcrafted (attention) | 1.52 | +290 |
| **Full MASD** | **0.39** | **-** |
| w/o adversarial | 1.52 | +290 |

**Key Finding:** Adversarial training provides the largest single improvement (1.52% -> 0.39%), confirming vocoder-agnostic learning is the most critical component.

---

## Cross-Dataset Zero-Shot Generalization (Fig. 2b)

| Dataset | Description | EER (%) |
|---------|------------|---------|
| FoR | 195K utterances, 10 TTS systems | 5.93 |
| ASVspoof 2021 DF | 182K utterances, codec conditions | 7.91 |
| WaveFake | 118K utterances, 6 vocoders | 6.42 |

---

## Confidence Calibration (Fig. 3)

| Metric | Uncalibrated | Temperature-Scaled |
|--------|-------------|-------------------|
| ECE | 15.1% | **1.1%** |
| Improvement | - | 93% |

**Selective Prediction (u=0.3):** 40% coverage at 1.85% EER. Rejected 60% contains 76% of errors.

**Uncertainty Awareness:** Correct mu=0.283 vs Incorrect mu=0.493.

---

## State-of-the-Art Comparison (Table III)

| Method | Framework | EER (%) |
|--------|-----------|---------|
| SAMO | Supervised | 1.09 |
| Wav2vec + AASIST2 | SSL | 1.61 |
| HuBert + WavLM | SSL | 0.42 |
| **MASD (Ours)** | **SSL** | **0.39** |

---

## Implementation Details

| Component | Specification |
|-----------|--------------|
| SSL Pretraining | 100 epochs, AdamW, lr=1e-4, batch=256, 4xRTX 3090, ~50h |
| Encoder | 5 ResBlocks (64->128->256->256->128), 3x3 kernel, stride 2 |
| CPC | 2-layer GRU, k=12 frames, 128 negatives, tau=0.07 |
| Masking | 15% of 16x16 patches, L2 reconstruction |
| Vocoders | WORLD, WaveGlow, MelGAN, HiFi-GAN, UnivNet, BigVGAN |
| Attention Module | 0.058M params, dk=dv=64, output 256-dim |
| Downstream | SGD lr=1e-3, 50 epochs, SVM retrained every 5 epochs |
| SVM | RBF kernel, C=10, gamma=0.01, class-weighted |
| Inference | 12ms/1s audio (CPU), 3ms/sample (GPU batch=32), 9.5MB model |
