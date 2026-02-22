"""
Evaluation Script for MASD (Section III-C).

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/masd/ \
        --data_dir data/ASVspoof2019/LA --split eval

    # Zero-shot cross-dataset
    python scripts/evaluate.py \
        --checkpoint experiments/masd/ \
        --data_dir data/WaveFake --dataset wavefake --zero_shot
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.masd import MASD
from models.calibrated_svm import CalibratedSVM
from datasets.asvspoof2019 import ASVspoof2019Dataset
from datasets.cross_dataset import FoRDataset, WaveFakeDataset, ASVspoof2021DFDataset
from utils.metrics import compute_eer, compute_ece


def extract_features(model, loader, device):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for specs, f_hand, labels in tqdm(loader, desc="Extracting"):
            feats = model(specs.to(device), f_hand.to(device))
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MASD")
    parser.add_argument("--checkpoint", type=str, required=True, help="Directory with model + SVM")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="asvspoof2019",
                        choices=["asvspoof2019", "for", "wavefake", "asvspoof2021df"])
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MASD(freeze_encoder=True).to(device)
    model_path = os.path.join(args.checkpoint, "masd_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    svm = CalibratedSVM()
    svm_path = os.path.join(args.checkpoint, "calibrated_svm.pkl")
    if os.path.exists(svm_path):
        svm.load(svm_path)

    # Build dataset
    if args.dataset == "asvspoof2019":
        dataset = ASVspoof2019Dataset(args.data_dir, split=args.split)
    elif args.dataset == "for":
        dataset = FoRDataset(args.data_dir)
    elif args.dataset == "wavefake":
        dataset = WaveFakeDataset(args.data_dir)
    elif args.dataset == "asvspoof2021df":
        dataset = ASVspoof2021DFDataset(args.data_dir)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    feats, labels = extract_features(model, loader, device)

    # Predictions
    preds, confs = svm.predict_with_confidence(feats)
    probs = svm.predict_proba(feats)

    acc = (preds == labels).mean() * 100
    eer = compute_eer(labels, probs[:, 1])
    ece = compute_ece(labels, probs[:, 1])

    mode = "Zero-Shot" if args.zero_shot else "In-Domain"
    print(f"\n{'='*60}")
    print(f"  MASD Evaluation | {args.dataset} | {mode}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.2f}%")
    print(f"  EER:         {eer:.4f}%")
    print(f"  ECE:         {ece:.4f}%")
    print(f"  Temperature: {svm.temperature:.4f}")
    print(f"{'='*60}\n")

    results = {"accuracy": acc, "eer": eer, "ece": ece, "dataset": args.dataset}
    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
