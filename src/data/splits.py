"""Identity-disjoint partitions for audio-visual forensic experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from src.data.dataloaders import AVClip


def identity_disjoint_splits(
    clips: Iterable[AVClip],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[AVClip]]:
    """Partition clips by speaker without identity leakage across splits."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three values that sum to 1.0")

    grouped: dict[str, list[AVClip]] = defaultdict(list)
    for clip in clips:
        grouped[clip.speaker_id].append(clip)
    speakers = sorted(grouped)
    if len(speakers) < 3:
        raise ValueError("at least three identities are required for train/val/test splits")

    shuffled = list(np.random.default_rng(seed).permutation(speakers))
    n = len(shuffled)
    n_train = max(1, round(n * ratios[0]))
    n_val = max(1, round(n * ratios[1]))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    split_speakers = {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train:n_train + n_val]),
        "test": set(shuffled[n_train + n_val:]),
    }
    return {
        name: [clip for speaker in speaker_ids for clip in grouped[speaker]]
        for name, speaker_ids in split_speakers.items()
    }
