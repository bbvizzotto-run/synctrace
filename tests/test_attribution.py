"""Unit tests for the SAE heatmap engine and localization metrics."""

import torch

from src.sae.spatiotemporal_attribution import (
    AttributionDecoder,
    LocalizationMetrics,
    SpatiotemporalAttributionEngine,
    auroc_simple,
    colorize_heatmap,
    frame_iou,
    make_region_mask,
    precision_at_k,
    region_iou,
)


def test_region_mask_shapes():
    m = make_region_mask(112, 112, "lips")
    assert m.shape == (112, 112)
    lips_top = int(112 * 0.15)
    assert m[lips_top + 5, 50] == 1.0
    assert m[5, 50] == 0.0


def test_colorize_heatmap_range():
    score = torch.rand(32, 32)
    img = colorize_heatmap(score)
    assert img.shape == (32, 32, 3)
    assert img.dtype == torch.uint8
    assert img.max() <= 255


def test_frame_iou_cases():
    pred = torch.tensor([0, 1, 2, 3])
    gt = torch.tensor([1, 2, 3, 4])
    assert abs(frame_iou(pred, gt) - 3 / 5) < 1e-9  # intersect {1,2,3}, union {0..4}
    assert frame_iou(pred, torch.tensor([])) == 0.0
    assert frame_iou(pred, pred) == 1.0


def test_region_iou():
    assert region_iou("lips", "lips") == 1.0
    assert region_iou("lips", "lower_face") == 0.0


def test_precision_at_k():
    scores = torch.tensor([0.9, 0.8, 0.1, 0.05, 0.0])
    gt = torch.tensor([1, 1, 0, 0, 1])
    assert abs(precision_at_k(scores, gt, k=2) - 1.0) < 1e-9
    assert precision_at_k(scores, gt, k=4) == 0.5  # 2 of 4 correct


def test_auroc_simple():
    scores = torch.tensor([0.9, 0.8, 0.3, 0.1])
    labels = torch.tensor([True, True, False, False])
    assert auroc_simple(scores, labels) == 1.0
    assert auroc_simple(torch.tensor([0.1, 0.2]),
                        torch.tensor([True, True])) != 0.0


def test_decoder_forward_shape():
    dec = AttributionDecoder(embed_dim=64, num_frames=8, num_regions=3,
                             num_bands=4)
    x = torch.randn(2, 64)
    logits, loss = dec(x)
    assert logits.shape == (2, 8, 7)  # 3 regions + 4 bands
    assert loss is None
    gt = torch.rand_like(logits)
    _, loss = dec(x, gt)
    assert loss is not None and loss.dim() == 0


def test_attribution_map_with_gt():
    """End-to-end SAE attribution using a real CML + SyncEncoder pipeline."""
    from src.cml.contrastive_misalignment_learner import ContrastiveMisalignmentLearner
    from src.encoder.sync_encoder import SyncEncoder
    from src.sae.spatiotemporal_attribution import AttributionDecoder

    torch.manual_seed(0)
    enc = SyncEncoder(embed_dim=64, stem_embed=32, num_mamba_blocks=1)
    model = ContrastiveMisalignmentLearner(enc, embed_dim=64)
    dec = AttributionDecoder(embed_dim=64, num_frames=16)
    engine = SpatiotemporalAttributionEngine(dec)

    audio = torch.randn(1, 80, 320, requires_grad=True)
    video = torch.randn(1, 16, 3, 112, 112, requires_grad=True)
    gt = torch.zeros(16, 112, 112)
    gt[8:16] = 1.0  # manipulated second half

    result = engine.attribution_map(model, audio, video, gt_mask=gt, )
    assert result["video_heatmap"].shape == (1, 16, 112, 112)
    assert result["frame_saliency"].shape == (1, 16)
    assert result["audio_band_scores"].shape[0] == 1
    assert result["audio_band_scores"].shape[2] == 4  # (B, T_mel, 4)
    assert "lips" in result["region_scores"]
    m = result["metrics"]
    assert isinstance(m, LocalizationMetrics)
    assert 0.0 <= m.precision_at_k <= 1.0
    assert m.miou_spatial in (0.0, 1.0)
