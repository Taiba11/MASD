"""
Downstream Training Script for MASD (Section III-B).

Encoder frozen; cross-attention module trained with SGD; SVM retrained every 5 epochs.
Temperature calibration applied after final SVM training.

Usage:
    python scripts/train.py \
        --encoder_ckpt checkpoints/ssl_pretrained/best_encoder.pth \
        --data_dir data/ASVspoof2019/LA \
        --output_dir experiments/masd
"""

import os
import sys
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.masd import MASD
from models.calibrated_svm import CalibratedSVM
from datasets.asvspoof2019 import ASVspoof2019Dataset


def extract_all_features(model, loader, device):
    """Extract fused features and labels for SVM training."""
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for specs, f_hand, labels in tqdm(loader, desc="Extracting features"):
            specs, f_hand = specs.to(device), f_hand.to(device)
            feats = model(specs, f_hand)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="MASD Downstream Training")
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/masd")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MASD Downstream Training")
    print(f"  Frozen encoder + Cross-attention + Calibrated SVM")
    print(f"{'='*60}\n")

    # Build model
    model = MASD(freeze_encoder=True).to(device)

    # Load pretrained encoder
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device)
        model.encoder.load_state_dict(ckpt["encoder"])
        print(f"Loaded encoder from {args.encoder_ckpt}")

    # Datasets
    train_dataset = ASVspoof2019Dataset(args.data_dir, split="train")
    dev_dataset = ASVspoof2019Dataset(args.data_dir, split="dev")

    # Split train into SVM-train (80%) and calibration (20%)
    n_cal = int(0.2 * len(train_dataset))
    n_train = len(train_dataset) - n_cal
    svm_train_ds, cal_ds = random_split(train_dataset, [n_train, n_cal])

    train_loader = DataLoader(svm_train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=4)

    # Train attention module
    attn_params = [p for p in model.attention.parameters() if p.requires_grad]
    attn_params += [model.band_weights]
    optimizer = torch.optim.SGD(attn_params, lr=args.lr)

    svm = CalibratedSVM(C=10, gamma=0.01)

    for epoch in range(1, args.epochs + 1):
        # Extract features and train SVM every 5 epochs
        if epoch % 5 == 1 or epoch == 1:
            feats, labels = extract_all_features(model, train_loader, device)
            svm.fit(feats, labels)

            # Evaluate on dev
            dev_feats, dev_labels = extract_all_features(model, dev_loader, device)
            dev_preds = svm.predict(dev_feats)
            acc = (dev_preds == dev_labels).mean() * 100
            print(f"  Epoch {epoch}: Dev Accuracy = {acc:.2f}%")

    # Final SVM training
    feats, labels = extract_all_features(model, train_loader, device)
    svm.fit(feats, labels)

    # Temperature calibration on held-out set
    cal_feats, cal_labels = extract_all_features(model, cal_loader, device)
    svm.calibrate(cal_feats, cal_labels)
    print(f"  Optimal temperature T = {svm.temperature:.4f}")

    # Save
    torch.save(model.state_dict(), os.path.join(args.output_dir, "masd_model.pth"))
    svm.save(os.path.join(args.output_dir, "calibrated_svm.pkl"))
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
