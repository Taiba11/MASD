"""
Single-File Inference for MASD.

Usage:
    python scripts/inference.py \
        --checkpoint experiments/masd/ \
        --audio_path path/to/audio.wav
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.masd import MASD
from models.calibrated_svm import CalibratedSVM
from models.handcrafted_features import HandcraftedFeatureExtractor
from datasets.preprocessing import AudioPreprocessor, LogMelExtractor


def main():
    parser = argparse.ArgumentParser(description="MASD Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MASD(freeze_encoder=True).to(device)
    model_path = os.path.join(args.checkpoint, "masd_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    svm = CalibratedSVM()
    svm.load(os.path.join(args.checkpoint, "calibrated_svm.pkl"))

    # Preprocess
    preprocessor = AudioPreprocessor()
    mel_extractor = LogMelExtractor()
    hand_extractor = HandcraftedFeatureExtractor()

    waveform = preprocessor.load(args.audio_path)
    spec = mel_extractor.extract(waveform).unsqueeze(0).to(device)
    f_hand = torch.from_numpy(hand_extractor.extract(waveform)).unsqueeze(0).float().to(device)

    # Extract features
    with torch.no_grad():
        f_fused = model(spec, f_hand).cpu().numpy()

    # Predict
    preds, confs = svm.predict_with_confidence(f_fused)
    probs = svm.predict_proba(f_fused)

    label = "REAL (Bonafide)" if preds[0] == 0 else "FAKE (Spoofed)"

    print(f"\n{'='*50}")
    print(f"  Prediction:  {label}")
    print(f"  Confidence:  {confs[0]:.4f}")
    print(f"  P(Real):     {probs[0, 0]:.4f}")
    print(f"  P(Fake):     {probs[0, 1]:.4f}")
    print(f"  Temperature: {svm.temperature:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
