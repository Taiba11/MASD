"""
Cross-Dataset Loaders for Zero-Shot Evaluation (Section III-A).

FoR: 195K utterances, 10 TTS systems
ASVspoof 2021 DF: 182K utterances, codec conditions
WaveFake: 118K utterances, 6 vocoders
"""

from pathlib import Path
from torch.utils.data import Dataset
import torch

from .preprocessing import AudioPreprocessor, LogMelExtractor
from models.handcrafted_features import HandcraftedFeatureExtractor

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


class _BaseDataset(Dataset):
    def __init__(self, file_list, sample_rate=16000, duration=4.0):
        self.file_list = file_list
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)
        self.mel_extractor = LogMelExtractor(sample_rate=sample_rate)
        self.hand_extractor = HandcraftedFeatureExtractor(sample_rate=sample_rate)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fp, label = self.file_list[idx]
        wav = self.preprocessor.load(fp)
        spec = self.mel_extractor.extract(wav)
        hand = torch.from_numpy(self.hand_extractor.extract(wav)).float()
        return spec, hand, torch.tensor(label, dtype=torch.long)


class FoRDataset(_BaseDataset):
    """FoR dataset for zero-shot cross-dataset evaluation."""

    def __init__(self, data_dir, **kwargs):
        files = []
        data_dir = Path(data_dir)
        for label_name, label in [("real", 0), ("fake", 1)]:
            for f in data_dir.rglob("*"):
                if f.suffix.lower() in AUDIO_EXTS and label_name in str(f).lower():
                    files.append((str(f), label))
        super().__init__(files, **kwargs)


class WaveFakeDataset(_BaseDataset):
    """WaveFake dataset for zero-shot cross-dataset evaluation."""

    def __init__(self, data_dir, **kwargs):
        files = []
        data_dir = Path(data_dir)
        for f in data_dir.rglob("*"):
            if f.suffix.lower() in AUDIO_EXTS:
                label = 0 if "real" in str(f.parent).lower() else 1
                files.append((str(f), label))
        super().__init__(files, **kwargs)


class ASVspoof2021DFDataset(_BaseDataset):
    """ASVspoof 2021 DF dataset for zero-shot evaluation."""

    def __init__(self, data_dir, protocol_path=None, **kwargs):
        files = []
        data_dir = Path(data_dir)
        if protocol_path and Path(protocol_path).exists():
            with open(protocol_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        name = parts[1]
                        label = 0 if parts[4] == "bonafide" else 1
                        fp = data_dir / "flac" / f"{name}.flac"
                        if fp.exists():
                            files.append((str(fp), label))
        else:
            for f in data_dir.rglob("*.flac"):
                files.append((str(f), 1))
        super().__init__(files, **kwargs)
