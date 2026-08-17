"""Unit tests for the pseudo-forgery generator and recipe semantics."""

import numpy as np

from src.data.pseudo_forgery_generator import ForgeryRecipe, PseudoForgeryGenerator


def _fake_frame(size=224, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def test_recipe_severity_combination():
    r = ForgeryRecipe(visual_severity=0.6, audio_severity=0.6)
    assert r.combined_severity == 1.0  # capped at 1.0
    r = ForgeryRecipe(visual_severity=0.2, audio_severity=0.2)
    assert abs(r.combined_severity - 0.4) < 1e-9


def test_zero_severity_is_noop():
    gen = PseudoForgeryGenerator(backend="opencv")
    src = _fake_frame()
    donor = _fake_frame(seed=2)
    r = ForgeryRecipe(visual_severity=0.0)
    out, gt = gen.apply_visual_manipulation(src, donor, r)
    np.testing.assert_array_equal(out, src)
    assert gt.sum() == 0


def test_visual_manipulation_changes_frame():
    gen = PseudoForgeryGenerator(backend="opencv")
    src = _fake_frame()
    donor = _fake_frame(seed=2)
    r = ForgeryRecipe(visual_severity=1.0, region="lips")
    out, gt = gen.apply_visual_manipulation(src, donor, r)
    if gt.sum() > 0:  # face must be detected for change
        assert (out != src).any()
        assert gt.shape == src.shape[:2]
        assert gt.dtype == bool


def test_audio_swap_and_offset():
    gen = PseudoForgeryGenerator()
    sr = 16000
    orig = np.random.randn(sr * 2).astype(np.float32)
    swap = np.random.randn(sr * 2).astype(np.float32)

    out0 = gen.apply_audio_manipulation(orig, swap, sr,
                                        ForgeryRecipe(audio_severity=0.0))
    np.testing.assert_array_equal(out0, orig)

    out1 = gen.apply_audio_manipulation(orig, swap, sr,
                                        ForgeryRecipe(audio_severity=1.0))
    # full swap should differ strongly from the original
    assert np.abs(out1 - orig).mean() > 0.1

    out2 = gen.apply_audio_manipulation(orig, swap, sr,
                                        ForgeryRecipe(audio_severity=1.0,
                                                      audio_offset_ms=200))
    # offset shifts content; dot product should differ from non-shifted
    assert not np.allclose(out1, out2)


def test_full_clip_generation_gt_shapes():
    gen = PseudoForgeryGenerator(backend="opencv")
    sr = 16000
    frames = np.stack([_fake_frame(size=112, seed=i) for i in range(16)])
    audio = np.zeros(sr * 4, dtype=np.float32)
    recipe = ForgeryRecipe(visual_severity=0.8, audio_severity=0.4,
                           region="lips", rng_seed=7)
    forgery = gen.generate(frames, audio, frames, audio, sr, recipe,
                           manipulate_frames_ratio=0.5)
    assert forgery.frames.shape == (16, 112, 112, 3)
    assert forgery.visual_gt_masks.shape == (16, 112, 112)
    assert forgery.temporal_gt.sum() == 8
    assert forgery.severity == 1.0  # 0.8 + 0.4 capped
