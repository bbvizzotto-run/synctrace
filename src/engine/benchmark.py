"""
SyncTrace — Benchmark & evaluation suite
==========================================
Runs the full experimental protocol for the paper:

  1. Severity ablation        — anomaly/estimation quality at severities
                                  s in {0.0, 0.25, 0.5, 0.75, 1.0}
  2. Detection evaluation     — AUC, EER, AP of the anomaly score
  3. Localization quality     — mIoU, Precision@K vs the automatic GT
  4. Efficiency               — FLOPs, params, latency for the paper table

Metrics follow the standard audio-visual forensics protocol (AUC, EER, AP
of the anomaly score), plus the proposed severity regression (MAE vs s)
and localization scores from the SAE.

Two execution modes:
  --mode synthetic  — controlled-severity pseudo-forgery synthesis on
                      randomly initialized frames (reproduces protocol
                      anywhere, no dataset download needed)
  --mode dataset    — real clips via SyncTraceDataset (FakeAVCeleb /
                      AV-LipSync-TIMIT loaders) once available
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.cml.contrastive_misalignment_learner import ContrastiveMisalignmentLearner
from src.data.pseudo_forgery_generator import ForgeryRecipe, PseudoForgeryGenerator
from src.encoder.sync_encoder import SyncEncoder, _flops_per_second
from src.sae.spatiotemporal_attribution import (
    AttributionDecoder,
    SpatiotemporalAttributionEngine,
)

H, W, FRAMES = 112, 112, 16
SAMPLE_RATE = 16000
N_SAMPLES = SAMPLE_RATE * 2  # 2-second clips


def _random_frames(n: int = FRAMES, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, H, W, 3)) * 255).astype(np.uint8)


def _random_audio(n: int = N_SAMPLES, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) * 2 - 1).astype(np.float32) * 0.5


def _mel(waveform: np.ndarray, sr: int = SAMPLE_RATE):
    """Import dataset mel helper to keep preprocessing identical."""
    from src.data.dataset import _mel_spectrogram
    return _mel_spectrogram(waveform, sr)


def make_pair(generator: PseudoForgeryGenerator, sev: float, seed: int):
    """One (authentic anchor, positive, negative-forgery) pair + GT mask."""
    rng = random.Random(seed)
    src_frames = _random_frames(seed=seed)
    src_audio = _random_audio(seed=seed + 1)
    donor_frames = _random_frames(seed=seed + 2)
    donor_audio = _random_audio(seed=seed + 3)
    recipe = ForgeryRecipe(
        visual_severity=sev,
        audio_severity=sev,
        audio_offset_ms=int(sev * 400),
        region="lips" if rng.random() < 0.5 else "lower_face",
        rng_seed=seed,
    )
    forgery = generator.generate(src_frames, src_audio,
                                 donor_frames, donor_audio,
                                 SAMPLE_RATE, recipe)
    return {
        "anchor_v": torch.from_numpy(src_frames).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0,
        "anchor_a": torch.from_numpy(_mel(src_audio)).float().unsqueeze(0),
        "pos_v": torch.from_numpy(src_frames).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0,
        "pos_a": torch.from_numpy(_mel(src_audio)).float().unsqueeze(0),
        "neg_v": torch.from_numpy(forgery.frames).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0,
        "neg_a": torch.from_numpy(_mel(forgery.audio)).float().unsqueeze(0),
        "severity": forgery.severity,
        "temporal_gt": torch.from_numpy(forgery.temporal_gt).float(),
        "visual_gt": torch.from_numpy(forgery.visual_gt_masks).float(),
        "region": forgery.region,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def eer_from_scores(positives: list, negatives: list, n_thr: int = 400):
    pos = torch.cat([s.flatten() for s in positives])
    neg = torch.cat([s.flatten() for s in negatives])
    thr = torch.linspace(min(pos.min(), neg.min()),
                         max(pos.max(), neg.max()), n_thr)
    far = (neg.unsqueeze(-1) > thr.unsqueeze(0)).float().mean(0)
    frr = (pos.unsqueeze(-1) <= thr.unsqueeze(0)).float().mean(0)
    return float((far - frr).abs().min())


def auroc_from_scores(positives: list, negatives: list):
    pos = torch.cat([s.flatten() for s in positives])
    neg = torch.cat([s.flatten() for s in negatives])
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    wins = (pos.unsqueeze(-1) > neg.unsqueeze(0)).float().mean()
    ties = (pos.unsqueeze(-1) == neg.unsqueeze(0)).float().mean() * 0.5
    return float(wins + ties)


def average_precision(positives: list, negatives: list):
    flat_pos = [s.flatten() for s in positives]
    flat_neg = [s.flatten() for s in negatives]
    scores = torch.cat(flat_pos + flat_neg)
    labels = torch.cat([torch.ones(s.numel()) for s in flat_pos]
                       + [torch.zeros(s.numel()) for s in flat_neg])
    order = scores.argsort(descending=True)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    tp = labels_sorted.cumsum(0)
    prec = tp / torch.arange(1, labels.numel() + 1)
    return float(prec[labels_sorted.bool()].mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-test", type=int, default=96)
    p.add_argument("--model-ckpt", default=None)
    p.add_argument("--mode", choices=["synthetic", "dataset"],
                   default="synthetic")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="quick contrastive warm-up when no checkpoint given")
    args, _ = p.parse_known_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg.get("model", {})
    embed_dim = mcfg.get("embed_dim", 256)
    device = torch.device(args.device)

    print(f"METRIC device={args.device}")
    print(f"METRIC mode={args.mode}")

    encoder = SyncEncoder(embed_dim=embed_dim,
                          stem_embed=mcfg.get("stem_embed", 64),
                          num_mamba_blocks=mcfg.get("num_mamba_blocks", 2))
    model = ContrastiveMisalignmentLearner(encoder, embed_dim=embed_dim)
    decoder = AttributionDecoder(embed_dim)
    engine = SpatiotemporalAttributionEngine(decoder)

    if args.model_ckpt and Path(args.model_ckpt).exists():
        state = torch.load(args.model_ckpt, map_location=device,
                           weights_only=True)
        model.load_state_dict(state.get("model", state), strict=False)

    # quick contrastive warm-up from synthetic pseudo-forgery pairs
    generator = PseudoForgeryGenerator(
        backend=cfg.get("data", {}).get("backend", "opencv"))

    if args.warmup_epochs > 0:
        opt = torch.optim.Adam(
            list(model.parameters()) + list(decoder.parameters()), lr=1e-3)
        for ep in range(args.warmup_epochs):
            losses = []
            for i in range(64):
                pair = make_pair(generator, sev=0.5, seed=3000 + i)
                pair = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in pair.items()}
                out = model(pair["anchor_a"], pair["anchor_v"],
                            pair["pos_a"], pair["pos_v"],
                            pair["neg_a"], pair["neg_v"],
                            torch.tensor([pair["severity"]], device=device))
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
                losses.append(out["loss"].item())
            print(f"METRIC warmup_epoch={ep} mean_loss={sum(losses)/len(losses):.4f}")
    model.eval()

    # -----------------------------------------------------------------
    # 1. severity ablation: anomaly + estimation + localization per sev
    # -----------------------------------------------------------------
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    sev_results = {}
    for sev in severities:
        mae_list, mious, patk_list = [], [], []
        anom_list = []
        for i in range(args.n_test):
            pair = make_pair(generator, sev, seed=1000 + i)
            pair = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in pair.items()}
            with torch.no_grad():
                out = model(pair["anchor_a"], pair["anchor_v"],
                            pair["pos_a"], pair["pos_v"],
                            pair["neg_a"], pair["neg_v"],
                            torch.zeros(1, device=device))
            anom_list.append(float(out["anomaly_score"].item()))
            est = float(model.severity_head(
                model.projector(encoder(pair["neg_a"], pair["neg_v"]))))
            mae_list.append(abs(est - sev))
            r = engine.attribution_map(model, pair["neg_a"], pair["neg_v"],
                                       gt_mask=pair["temporal_gt"])
            mious.append(r["metrics"].miou_temporal)
            patk_list.append(r["metrics"].precision_at_k)
        sev_results[str(sev)] = {
            "anomaly_mean": float(np.mean(anom_list)),
            "severity_mae": float(np.mean(mae_list)),
            "localization_miou": float(np.mean(mious)),
            "precision_at_k": float(np.mean(patk_list)),
        }
        print(f"METRIC sev_{sev}_anomaly_mean={sev_results[str(sev)]['anomaly_mean']:.4f}")
        print(f"METRIC sev_{sev}_severity_mae={sev_results[str(sev)]['severity_mae']:.4f}")
        print(f"METRIC sev_{sev}_localization_miou={sev_results[str(sev)]['localization_miou']:.4f}")
        print(f"METRIC sev_{sev}_precision_at4={sev_results[str(sev)]['precision_at_k']:.4f}")

    # -----------------------------------------------------------------
    # 2. detection metrics: sev=1.0 positives vs sev=0.0 negatives
    # -----------------------------------------------------------------
    pos_scores, neg_scores = [], []
    for i in range(args.n_test):
        for sev, bucket in ((1.0, pos_scores), (0.0, neg_scores)):
            pair = make_pair(generator, sev, seed=2000 + i)
            pair = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in pair.items()}
            with torch.no_grad():
                out = model(pair["anchor_a"], pair["anchor_v"],
                            pair["pos_a"], pair["pos_v"],
                            pair["neg_a"], pair["neg_v"],
                            torch.zeros(1, device=device))
            bucket.append(out["anomaly_score"].detach())
    auc = auroc_from_scores(pos_scores, neg_scores)
    eer = eer_from_scores(pos_scores, neg_scores)
    ap = average_precision(pos_scores, neg_scores)
    print(f"METRIC auc={auc:.4f}")
    print(f"METRIC eer={eer:.4f}")
    print(f"METRIC average_precision={ap:.4f}")

    # -----------------------------------------------------------------
    # 3. efficiency table for the paper
    # -----------------------------------------------------------------
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(10):
            encoder(torch.randn(1, 80, 320, device=device),
                    torch.randn(1, 16, 3, 112, 112, device=device))
        latency = (time.perf_counter() - t0) / 10
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"METRIC latency_ms={latency * 1000:.2f}")
    print(f"METRIC flops_gflops_25fps={_flops_per_second(1e6, fps=25.0) / 1e9:.2f}")
    print(f"METRIC params_m={n_params:.2f}")

    # -----------------------------------------------------------------
    # 4. persist results
    # -----------------------------------------------------------------
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    report = {
        "severity_ablation": sev_results,
        "detection": {"auc": auc, "eer": eer, "average_precision": ap},
        "efficiency": {"latency_ms": latency * 1000,
                       "flops_gflops_25fps": _flops_per_second(1e6, 25.0) / 1e9,
                       "params_m": n_params},
        "config": vars(args),
    }
    (out / "benchmark_results.json").write_text(json.dumps(report, indent=2))
    print("METRIC benchmark_saved=experiments/benchmark_results.json")


if __name__ == "__main__":
    main()
