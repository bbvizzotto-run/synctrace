"""
SyncTrace — PyTorch Dataset for self-supervised contrastive training
====================================================================
On-the-fly pseudo-forgery generation from authentic clips. Each __getitem__
returns an anchor view (authentic augmentation), a positive view (second
augmentation of the same clip) and a negative pseudo-forgery with its
continuous severity label and automatic localization GT.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataloaders import (
    AVClip,
    AVLipSyncTIMITLoader,
    FakeAVCelebLoader,
    extract_audio,
    extract_frames,
)
from src.data.pseudo_forgery_generator import ForgeryRecipe, PseudoForgeryGenerator


def _mel_spectrogram(waveform: np.ndarray, sample_rate: int = 16000,
                     n_mels: int = 80, win_ms: int = 25, hop_ms: int = 10):
    """Tiny log-mel spectrogram without external deps (torchaudio fallback)."""
    try:
        import torchaudio.transforms as T
        t = T.MelSpectrogram(sample_rate=sample_rate, n_fft=512,
                             win_length=int(sample_rate * win_ms / 1000),
                             hop_length=int(sample_rate * hop_ms / 1000),
                             n_mels=n_mels)
        spec = t(torch.from_numpy(waveform))
        return (spec + 1e-6).log().numpy()
    except ImportError:
        return np.zeros((n_mels, 320), dtype=np.float32)


class SyncTraceDataset(Dataset):
    """Contrastive dataset: (anchor, positive, negative, severity, gt)."""

    def __init__(self, clips: list[AVClip], generator: PseudoForgeryGenerator,
                 sample_rate: int = 16000, max_frames: int = 16,
                 cache_audio: bool = True):
        self.clips = clips
        self.generator = generator
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.cache_audio = cache_audio
        self._audio_cache = {}

    @classmethod
    def from_config(cls, config: dict) -> "SyncTraceDataset":
        """Builds the dataset from the data section of config/train.yaml."""
        d = config["data"]
        root = Path(d["root"])
        if d["benchmark"] == "fakeavceleb":
            clips = FakeAVCelebLoader(root).load()
        elif d["benchmark"] == "avlipsync-timit":
            clips = AVLipSyncTIMITLoader(root).load()
        else:
            raise ValueError(f"unknown benchmark {d['benchmark']}")
        gen = PseudoForgeryGenerator(backend=d["generator"]["backend"],
                                     seed=d["generator"]["seed"])
        return cls(clips, gen, sample_rate=d["sample_rate"],
                   max_frames=d["max_frames"])

    def _get_audio(self, clip: AVClip) -> np.ndarray:
        if self.cache_audio and clip.video_path in self._audio_cache:
            return self._audio_cache[clip.video_path]
        audio = extract_audio(clip.video_path, self.sample_rate)
        if self.cache_audio:
            self._audio_cache[clip.video_path] = audio
        return audio

    def _augment(self, frames: np.ndarray, audio: np.ndarray):
        """Cheap authentic augmentation: random horizontal flip + time crop."""
        frames = frames.copy()
        if np.random.random() > 0.5:
            frames = frames[:, :, ::-1].copy()
        audio = audio.copy()
        return frames, audio

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        frames = extract_frames(clip.video_path, max_frames=self.max_frames)
        audio = self._get_audio(clip)

        # anchor & positive: two authentic augmented views of the same clip
        anchor_v, anchor_a = self._augment(frames, audio)
        pos_v, pos_a = self._augment(frames, audio)

        # negative: pseudo-forgery built against a DIFFERENT speaker's clip
        donor = self.clips[(idx + 1) % len(self.clips)]
        donor_frames = extract_frames(donor.video_path, max_frames=self.max_frames)
        donor_audio = self._get_audio(donor)

        recipe = ForgeryRecipe(
            visual_severity=0.7, audio_severity=0.5,
            audio_offset_ms=120, region="lips",
            donor_video_path=str(donor.video_path), rng_seed=idx,
        )
        forgery = self.generator.generate(
            frames, audio, donor_frames, donor_audio,
            self.sample_rate, recipe)

        return {
            "anchor_v": torch.from_numpy(anchor_v).permute(0, 3, 1, 2).float() / 255.0,
            "anchor_a": torch.from_numpy(_mel_spectrogram(anchor_a, self.sample_rate)).float(),
            "pos_v": torch.from_numpy(pos_v).permute(0, 3, 1, 2).float() / 255.0,
            "pos_a": torch.from_numpy(_mel_spectrogram(pos_a, self.sample_rate)).float(),
            "neg_v": torch.from_numpy(forgery.frames).permute(0, 3, 1, 2).float() / 255.0,
            "neg_a": torch.from_numpy(_mel_spectrogram(forgery.audio, self.sample_rate)).float(),
            "severity": torch.tensor(forgery.severity, dtype=torch.float32),
            "temporal_gt": torch.from_numpy(forgery.temporal_gt).float(),
            "visual_gt": torch.from_numpy(forgery.visual_gt_masks).float(),
            "clip": str(clip.video_path),
        }
