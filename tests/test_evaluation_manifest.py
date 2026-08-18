"""Tests for the explicit, auditable benchmark-manifest contract."""

import csv

import pytest

from src.engine.evaluate import load_test_manifest


def test_manifest_loads_labeled_clips_under_dataset_root(tmp_path):
    (tmp_path / "real.mp4").touch()
    manifest = tmp_path / "test.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "label", "speaker_id", "source"])
        writer.writeheader()
        writer.writerow({"relative_path": "real.mp4", "label": "real", "speaker_id": "s1", "source": "test"})
    rows = load_test_manifest(str(manifest), str(tmp_path))
    assert rows[0]["label"] == "real"


def test_manifest_rejects_paths_outside_dataset_root(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("relative_path,label,speaker_id,source\n../outside.mp4,fake,s1,test\n")
    with pytest.raises(FileNotFoundError):
        load_test_manifest(str(manifest), str(tmp_path))
