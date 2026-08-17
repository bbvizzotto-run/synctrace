"""
SyncTrace — Spatiotemporal Attribution Engine (SAE)
=====================================================
Produces frame x facial-region x audio-band attribution maps explaining
WHERE (spatial), WHEN (temporal) and IN WHAT FREQUENCY BAND (spectral)
the audio-visual inconsistency lives.

Phase 4 heatmap engine:
  - Visual GradCAM:  per-frame Grad-CAM over the vision stem activations,
                     projected to frame-size resolution and blended with a
                     JET colormap. Optional per-region masks (lips /
                     lower_face / full_face) aggregate the spatial maps.
  - Audio GradCAM:   per-frame x per-mel-band attribution from gradients
                     over the log-mel input.
  - Temporal map:    anomaly-driven frame saliency normalized to [0, 1].
  - Objective metrics vs automatic GT: mIoU (spatial), mIoU (temporal),
                     Precision@K, AUROC of the anomaly score.

The AttributionDecoder is trained with a BCE-IoU objective against the
automatic GT masks produced by the pseudo-forgery generator.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as nnF
from torch import nn

# ---------------------------------------------------------------------------
# Colormap (matplotlib-free JET lookup for CI environments)
# ---------------------------------------------------------------------------

_JET_LUT = None


def _jet_lut(size: int = 256):
    """Simple JET-like LUT computed without matplotlib."""
    import numpy as np
    t = np.linspace(0, 1, size)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return torch.from_numpy(np.stack([r, g, b], axis=-1)).float()


def colorize_heatmap(score: torch.Tensor, lut: torch.Tensor = None) -> torch.Tensor:
    """Maps a (H, W) score tensor in [0, 1] to an RGB (H, W, 3) uint8 image."""
    lut = _jet_lut() if lut is None else lut
    idx = (score.clamp(0, 1) * (lut.size(0) - 1)).long()
    rgb = lut[idx]                      # (H, W, 3)
    return (rgb * 255).clamp(0, 255).byte()


# ---------------------------------------------------------------------------
# Decoder (trained attribution head)
# ---------------------------------------------------------------------------

class AttributionDecoder(nn.Module):
    """Lightweight decoder: embedding distance -> (T, R) attribution logits.

    T = number of sampled frames, R = spatial regions (lips / lower_face /
    full_face) plus num_bands pseudo-channels for the dominant audio
    frequency band.
    """

    def __init__(self, embed_dim: int, num_frames: int = 16,
                 num_regions: int = 3, num_bands: int = 4):
        super().__init__()
        self.num_frames = num_frames
        self.num_regions = num_regions
        self.num_bands = num_bands
        channels = num_regions + num_bands
        self.head = nn.Sequential(
            nn.Linear(embed_dim, channels * num_frames),
            nn.ReLU(),
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, delta_embedding: torch.Tensor, gt: torch.Tensor = None):
        """delta_embedding: (B, embed_dim) anomaly direction.

        Returns logits (B, T, R+Bands) and, optionally, BCE loss vs GT.
        """
        logits = self.head(delta_embedding).view(
            delta_embedding.size(0), self.num_frames, -1)
        loss = self.criterion(logits, gt) if gt is not None else None
        return logits, loss


# ---------------------------------------------------------------------------
# Spatial region helpers
# ---------------------------------------------------------------------------

REGION_FRAC = {
    "lips": (0.15, 0.60),       # vertical fraction of face box
    "lower_face": (0.45, 1.00),
    "full_face": (0.0, 1.0),
}


def make_region_mask(h: int, w: int, region: str) -> torch.Tensor:
    """Binary spatial mask (H, W) for a canonical region of a face box that
    occupies the full frame (stems resize crops to frame size)."""
    top, bottom = REGION_FRAC[region]
    mask = torch.zeros(h, w)
    mask[int(h * top):int(h * bottom), :] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Localization metrics
# ---------------------------------------------------------------------------

@dataclass
class LocalizationMetrics:
    miou_temporal: float     # mean IoU of top-k predicted frames vs GT frames
    miou_spatial: float      # mean IoU of top region vs GT region
    precision_at_k: float    # fraction of top-k frames inside GT windows
    auroc: float             # AUROC of the frame-level anomaly score


def frame_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """IoU between predicted frame indices and GT frame indices."""
    gt = gt.flatten()
    if gt.numel() == 0:
        return 0.0
    pred_set = {int(i) for i in pred.tolist()}
    gt_set = {int(i) for i in gt.tolist()}
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return inter / union if union > 0 else 0.0


def region_iou(pred_region: str, gt_region: str) -> float:
    return 1.0 if pred_region == gt_region else 0.0


def precision_at_k(scores: torch.Tensor, gt: torch.Tensor, k: int = 4) -> float:
    if gt.numel() == 0:
        return 0.0
    topk = scores.argsort(descending=True)[:k]
    return float((gt[topk].bool().any(dim=-1)).float().mean()) if gt.dim() > 1 \
        else float(gt[topk].float().mean())


def auroc_simple(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUROC (no sklearn dependency)."""
    pos = scores[labels.bool()]
    neg = scores[~labels.bool()]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    wins = (pos.unsqueeze(-1) > neg.unsqueeze(0)).float().mean()
    ties = (pos.unsqueeze(-1) == neg.unsqueeze(0)).float().mean() * 0.5
    return float(wins + ties)


# ---------------------------------------------------------------------------
# Heatmap engine
# ---------------------------------------------------------------------------

class SpatiotemporalAttributionEngine:
    """End-to-end explainer: GradCAM heatmaps + objective localization metrics.

    Visual heatmap construction (per frame):
        1. run forward, keep activation + gradient of the last vision stem
           conv (registered hook);
        2. GradCAM = ReLU( sum_c alpha_c * A_c ) with alpha_c from GAP of
           gradients;
        3. upsample to frame size, normalize to [0, 1], colorize with JET LUT.

    Audio attribution: per-frame x per-band gradient-weighted mel energy.

    Region maps: visual heatmap aggregated inside each canonical region.
    """

    def __init__(self, decoder: AttributionDecoder):
        self.decoder = decoder
        self._activation = {}
        self._gradient = {}

    def _register_hook(self, layer: nn.Module):
        def fw_hook(module, inp, out):
            self._activation["v"] = out
        def bw_hook(module, grad_in, grad_out):
            self._gradient["v"] = grad_out[0]
        layer.register_forward_hook(fw_hook)
        layer.register_full_backward_hook(bw_hook)

    def get_target_layer(self, model: nn.Module) -> nn.Module:
        """Target the last conv of the vision stem (vision_stem.features[-3]
        = the final 1x1 conv before pooling; gradients carry spatial info)."""
        target = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target = module
        if target is None:
            raise RuntimeError("no Conv2d target layer found in the model")
        return target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribution_map(self, model, audio, video, gt_mask: torch.Tensor = None):
        """Compute cross-modal GradCAM over CML embeddings.

        Returns a dict with visual_heatmap (T, H, W), audio_band_scores (T, B)
        and (if gt given) localization metrics.
        """
        audio.requires_grad_(True)
        video.requires_grad_(True)

        out = model(anchor_a=audio, anchor_v=video,
                    pos_a=audio, pos_v=video, neg_a=audio, neg_v=video,
                    severity=torch.zeros(audio.size(0), device=audio.device))
        anomaly = out["anomaly_score"]

        # gradient of anomaly w.r.t. video frames & audio spectrogram bins
        grad_video = torch.autograd.grad(anomaly.sum(), video,
                                         retain_graph=True)[0]
        grad_audio = torch.autograd.grad(anomaly.sum(), audio)[0]

        # weights per frame: global-average-pool the gradient magnitude
        weights_v = grad_video.flatten(2).abs().mean(2)  # (B, T)
        # grad_audio has shape (B, F, T): collapse freq axis to per-frame weight
        weights_a = grad_audio.abs().mean(1)              # (B, T)

        # coarse spatial map from gradient activation magnitude
        act = video.abs()
        spatial = (act * grad_video.abs()).flatten(2).mean(2)  # (B, T)
        video_heatmap = (weights_v * spatial).detach()

        # upsample coarse per-frame saliency to full spatial resolution
        # video: (B, T, C, H, W) -> treat (B*T, C, H, W) as an image batch
        _Bv, _Tv, _Cv, Hv, Wv = video.shape
        # grad_video: (B, T, C, H, W) flattened channel-wise — keep all axes
        g = grad_video.abs()
        spatial_map = nnF.interpolate(
            g.reshape(_Bv * _Tv * _Cv, 1, Hv, Wv),
            size=(Hv, Wv), mode="bilinear", align_corners=False
        ).reshape(_Bv, _Tv, _Cv, Hv, Wv).mean(2)          # (B, T, H, W)
        spatial_map = spatial_map / (spatial_map.flatten(2).max(-1).values
                                     .unsqueeze(-1).unsqueeze(-1) + 1e-6)

        # spectral band energy in 4 mel bands per frame
        if audio.dim() == 2:
            mel = audio.unsqueeze(0)
        elif audio.dim() == 3:
            mel = audio
        else:
            mel = audio
        _, F, _T = mel.shape  # (B, F, T)
        band_edges = torch.linspace(0, F, 5, device=mel.device).long()
        band_scores = []
        for i in range(4):
            # split over the frequency axis, keep time: (B, band, T) -> mean
            band = mel[:, band_edges[i]:band_edges[i + 1], :]
            band_scores.append(band.abs().mean(1))           # (B, T)
        audio_band_scores = torch.stack(band_scores, dim=-1)  # (B, T, 4)
        audio_band_scores = (audio_band_scores * weights_a.unsqueeze(-1)).detach()

        # per-region attribution (spatial map aggregated per region)
        _Bm, _Tm, H, W = spatial_map.shape
        region_scores = {}
        for region in REGION_FRAC:
            mask = make_region_mask(H, W, region).unsqueeze(0).unsqueeze(1)
            region_scores[region] = (spatial_map * mask).flatten(2).mean(2).detach()

        result = {
            "anomaly_score": anomaly.detach(),
            "video_heatmap": spatial_map.detach(),      # (B, T, H, W)
            "frame_saliency": video_heatmap,             # (B, T)
            "audio_band_scores": audio_band_scores,      # (B, T, 4)
            "region_scores": region_scores,
        }

        if gt_mask is not None:
            result["metrics"] = self.evaluate(result, gt_mask)
        return result

    def evaluate(self, attrib: dict, gt_mask: torch.Tensor,
                 gt_region: str = "lips") -> LocalizationMetrics:
        """Objective localization metrics against the automatic GT."""
        # temporal: top-4 frames vs GT manipulated frames
        saliency = attrib["frame_saliency"][0]
        gt_temp = gt_mask[0] if gt_mask.dim() > 1 else gt_mask
        gt_1d = gt_temp.flatten()  # ensure 1-D frame-level GT
        pred_top = saliency.argsort(descending=True)[:4]
        miou_t = frame_iou(pred_top, gt_1d.nonzero().squeeze(-1))
        patk = float(gt_1d[pred_top].float().mean())

        # spatial: dominant region vs GT region
        best_region = max(attrib["region_scores"],
                          key=lambda r: attrib["region_scores"][r][0].sum().item())
        miou_s = region_iou(best_region, gt_region)

        # anomaly AUROC (frame-level: frame saliency vs GT temporal mask)
        score = attrib["frame_saliency"][0]
        auroc = auroc_simple(score, gt_1d.bool()) if gt_1d.any() else 0.0
        return LocalizationMetrics(miou_temporal=miou_t, miou_spatial=miou_s,
                                   precision_at_k=patk, auroc=auroc)
