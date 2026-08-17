"""Unit tests for recipe grids and dataset integration."""

import numpy as np

from src.data.pseudo_forgery_generator import PseudoForgeryGenerator


def test_default_recipe_grid_coverage():
    recipes = PseudoForgeryGenerator.default_recipe_grid()
    severities = {round(r.visual_severity + r.audio_severity, 3)
                  for r in recipes}
    assert 0.0 in severities
    assert any(s >= 1.0 for s in severities)  # combined max-severity recipe
    assert all(0.0 <= r.combined_severity <= 1.0 for r in recipes)


def test_dataset_synthetic_sample_shape(tmp_path):
    """Dataset __getitem__ works end-to-end on synthetic AVClip stubs."""

    # stub clip whose video path is a generated mp4
    import cv2

    from src.data.dataloaders import AVClip
    from src.data.dataset import SyncTraceDataset

    video_path = tmp_path / "stub.mp4"
    frames = [(np.random.randint(0, 255, (112, 112, 3), np.uint8))
              for _ in range(24)]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (112, 112))
    for f in frames:
        writer.write(f)
    writer.release()

    clips = [AVClip(video_path=video_path, audio_path=video_path,
                    speaker_id="s1", metadata={"benchmark": "test"})]
    ds = SyncTraceDataset(clips, PseudoForgeryGenerator(backend="opencv"),
                          cache_audio=False)
    item = ds[0]
    assert item["anchor_v"].shape == (16, 3, 112, 112)
    assert item["neg_v"].shape == (16, 3, 112, 112)
    assert 0.0 <= float(item["severity"]) <= 1.0
    assert item["visual_gt"].shape == (16, 112, 112)
    assert item["temporal_gt"].shape == (16,)
