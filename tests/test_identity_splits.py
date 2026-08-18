"""Tests for leakage-resistant experimental partitions."""

from pathlib import Path

import pytest

from src.data.dataloaders import AVClip
from src.data.splits import identity_disjoint_splits


def _clip(speaker: str, index: int) -> AVClip:
    path = Path(f"/tmp/{speaker}_{index}.mp4")
    return AVClip(path, path, speaker, {})


def test_identity_splits_are_reproducible_and_disjoint():
    clips = [_clip(f"speaker_{speaker}", index)
             for speaker in range(10) for index in range(2)]
    first = identity_disjoint_splits(clips, seed=42)
    second = identity_disjoint_splits(clips, seed=42)
    first_ids = {name: {clip.speaker_id for clip in group} for name, group in first.items()}
    assert first_ids == {name: {clip.speaker_id for clip in group} for name, group in second.items()}
    assert first_ids["train"].isdisjoint(first_ids["val"])
    assert first_ids["train"].isdisjoint(first_ids["test"])
    assert first_ids["val"].isdisjoint(first_ids["test"])


def test_identity_splits_require_enough_identities():
    with pytest.raises(ValueError, match="three identities"):
        identity_disjoint_splits([_clip("a", 0), _clip("b", 0)])
