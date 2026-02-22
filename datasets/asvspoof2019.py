"""
ASVspoof 2019 LA Dataset Loader for MASD (Section III-A).

Train: 25,380 (2,580 real + 22,800 fake from A01-A06)
Dev: 24,844
Eval: 71,237 (unseen attacks A07-A19)
"""

import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

from .preprocessing import AudioPreprocessor, LogMelExtractor
from models.handcrafted_features import HandcraftedFeatureExtractor


class ASVspoof2019Dataset(Dataset):
    """
    ASVspoof 2019 LA dataset.

    Args:
        data_dir (str): Root LA directory.
        split (str): 'train', 'dev', or 'eval'.
        sample_rate (int): Audio sample rate.
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1}
    SPLITS = {
        "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
        "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
        "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
    }

    def __init__(self, data_dir, split="train", sample_rate=16000, duration=4.0):
        self.data_dir = Path(data_dir)
        audio_subdir, protocol_name = self.SPLITS[split]
        self.audio_dir = self.data_dir / audio_subdir / "flac"
        protocol_path = self.data_dir / "ASVspoof2019_LA_cm_protocols" / protocol_name

        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)
        self.mel_extractor = LogMelExtractor(sample_rate=sample_rate)
        self.hand_extractor = HandcraftedFeatureExtractor(sample_rate=sample_rate)

        self.file_list = self._parse_protocol(protocol_path)

    def _parse_protocol(self, protocol_path):
        items = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    name, label_str = parts[1], parts[4]
                    fp = self.audio_dir / f"{name}.flac"
                    if fp.exists():
                        items.append((str(fp), self.LABEL_MAP.get(label_str, 1)))
        return items

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]
        waveform = self.preprocessor.load(filepath)
        spectrogram = self.mel_extractor.extract(waveform)
        f_hand = torch.from_numpy(self.hand_extractor.extract(waveform)).float()
        return spectrogram, f_hand, torch.tensor(label, dtype=torch.long)
