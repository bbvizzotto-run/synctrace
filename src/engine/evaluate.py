"""Evaluation for licensed SyncTrace experiments.

The test manifest is deliberately explicit: it prevents assumptions about a
dataset archive layout and makes every real/fake label, speaker, and source
auditable. Pseudo-forgeries on held-out authentic identities supply severity
and localization ground truth; real benchmark clips supply detection metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.cml.contrastive_misalignment_learner import ContrastiveMisalignmentLearner
from src.data.dataloaders import extract_audio, extract_frames
from src.data.dataset import _mel_spectrogram
from src.encoder.sync_encoder import SyncEncoder
from src.engine.benchmark import auroc_from_scores, average_precision, eer_from_scores
from src.engine.train import build_real_loaders
from src.sae.spatiotemporal_attribution import (
    AttributionDecoder,
    frame_iou,
    precision_at_k,
)


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_test_manifest(path: str, data_root: str) -> list[dict[str, str]]:
    """Load an auditable CSV with relative_path,label,speaker_id,source columns."""
    required = {"relative_path", "label", "speaker_id", "source"}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest requires columns: {', '.join(sorted(required))}")
    root = Path(data_root).resolve()
    for row in rows:
        if row["label"] not in {"real", "fake"}:
            raise ValueError("manifest labels must be 'real' or 'fake'")
        clip = (root / row["relative_path"]).resolve()
        if not clip.is_relative_to(root) or not clip.exists():
            raise FileNotFoundError(f"manifest clip is missing or outside root: {row['relative_path']}")
        row["absolute_path"] = str(clip)
    return rows


def _build_model(config: dict[str, Any], checkpoint: str, device: torch.device):
    model_config = config["model"]
    encoder = SyncEncoder(
        embed_dim=model_config["embed_dim"], stem_embed=model_config.get("stem_embed", 128),
        num_mamba_blocks=model_config.get("num_mamba_blocks", 2),
        d_state=model_config.get("d_state", 16), keep_ratio=model_config.get("keep_ratio", 0.5),
        frame_size=model_config["frame_size"],
    )
    model = ContrastiveMisalignmentLearner(
        encoder, embed_dim=model_config["embed_dim"], projection_dim=model_config["projection_dim"])
    decoder = AttributionDecoder(model_config["projection_dim"], num_frames=model_config["num_frames"])
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    decoder.load_state_dict(state["attribution"])
    return model.to(device).eval(), decoder.to(device).eval()


def _clip_tensors(path: str, config: dict[str, Any], device: torch.device):
    data = config["data"]
    frames = extract_frames(Path(path), max_frames=data["max_frames"], target_size=config["model"]["frame_size"])
    audio = extract_audio(Path(path), sample_rate=data["sample_rate"])
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0
    mel = torch.from_numpy(_mel_spectrogram(audio, data["sample_rate"])).float().unsqueeze(0).to(device)
    return mel, video


def _reference_embedding(model, loader, device: torch.device) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.encode(batch["anchor_a"].to(device), batch["anchor_v"].to(device)))
    reference = torch.cat(embeddings).mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(reference, dim=-1)


def _pseudo_metrics(model, decoder, loader, device: torch.device) -> dict[str, float]:
    severity_errors, temporal_iou, temporal_patk = [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["anchor_a"].to(device), batch["anchor_v"].to(device),
                batch["pos_a"].to(device), batch["pos_v"].to(device),
                batch["neg_a"].to(device), batch["neg_v"].to(device), batch["severity"].to(device),
            )
            severity_errors.extend((output["severity_pred"] - batch["severity"].to(device)).abs().cpu().tolist())
            logits, _ = decoder(output["anomaly_direction"])
            temporal_scores = logits.sigmoid().amax(dim=-1).cpu()
            for score, target in zip(temporal_scores, batch["temporal_gt"]):
                temporal_iou.append(frame_iou(score, target))
                temporal_patk.append(precision_at_k(score, target, k=min(4, score.numel())))
    return {
        "severity_mae": float(np.mean(severity_errors)),
        "temporal_miou": float(np.mean(temporal_iou)),
        "temporal_precision_at_4": float(np.mean(temporal_patk)),
    }


def _efficiency(model, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    audio = torch.randn(1, 80, 320, device=device)
    video = torch.randn(1, config["model"]["num_frames"], 3, config["model"]["frame_size"], config["model"]["frame_size"], device=device)
    with torch.no_grad():
        for _ in range(10):
            model.encode(audio, video)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(30):
            start = time.perf_counter()
            model.encode(audio, video)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "flops_g": model.encoder.flops_estimate(config["model"]["num_frames"]) / 1e9,
        "latency_ms_median": float(np.median(timings)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/campaign.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="experiments/reports/evaluation.json")
    args = parser.parse_args()
    config = _load_config(args.config)
    device = torch.device(args.device)
    model, decoder = _build_model(config, args.checkpoint, device)
    loaders = build_real_loaders(config, args.data_root, args.seed)
    reference = _reference_embedding(model, loaders["train"], device)
    real_scores, fake_scores = [], []
    rows = load_test_manifest(args.test_manifest, args.data_root)
    with torch.no_grad():
        for row in rows:
            audio, video = _clip_tensors(row["absolute_path"], config, device)
            score = torch.pairwise_distance(model.encode(audio, video), reference).cpu()
            (real_scores if row["label"] == "real" else fake_scores).append(score)
    if not real_scores or not fake_scores:
        raise ValueError("test manifest must contain at least one real and one fake clip")
    report = {
        "status": "licensed_dataset_result",
        "dataset": config["data"]["benchmark"], "seed": args.seed,
        "manifest": str(Path(args.test_manifest).resolve()),
        "detection": {
            "auroc": auroc_from_scores(fake_scores, real_scores),
            "eer": eer_from_scores(fake_scores, real_scores),
            "average_precision": average_precision(fake_scores, real_scores),
            "n_real": len(real_scores), "n_fake": len(fake_scores),
        },
        "pseudo_forgery": _pseudo_metrics(model, decoder, loaders["test"], device),
        "efficiency": _efficiency(model, config, device),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"METRIC mode=licensed_dataset auroc={report['detection']['auroc']:.4f}")
    print(f"METRIC report={target}")


if __name__ == "__main__":
    main()
