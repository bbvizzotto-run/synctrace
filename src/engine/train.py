"""SyncTrace training entry point for CI smoke runs and licensed datasets.

Without ``--data-root`` this module intentionally performs a synthetic smoke
run. With a root of an approved dataset it creates identity-disjoint splits,
trains on authentic clips plus on-the-fly pseudo-forgeries, and selects a
checkpoint using validation loss only. Smoke metrics are never benchmark data.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.cml.contrastive_misalignment_learner import ContrastiveMisalignmentLearner
from src.data.dataloaders import AVLipSyncTIMITLoader, FakeAVCelebLoader
from src.data.dataset import SyncTraceDataset
from src.data.pseudo_forgery_generator import PseudoForgeryGenerator
from src.data.splits import identity_disjoint_splits
from src.encoder.sync_encoder import SyncEncoder
from src.sae.spatiotemporal_attribution import AttributionDecoder


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _make_dataset(config: dict[str, Any], clips, seed: int) -> SyncTraceDataset:
    data = config["data"]
    generator_config = data["generator"]
    generator = PseudoForgeryGenerator(
        backend=generator_config["backend"], seed=seed)
    return SyncTraceDataset(
        clips, generator, sample_rate=data["sample_rate"],
        max_frames=data["max_frames"], severity_values=tuple(generator_config["severities"]),
        regions=tuple(generator_config["regions"]),
        manipulate_frames_ratio=generator_config["manipulate_frames_ratio"],
        seed=seed,
    )


def build_real_loaders(config: dict[str, Any], root: str, seed: int) -> dict[str, DataLoader]:
    """Build train/validation/test loaders for an officially accessible root."""
    data = config["data"]
    benchmark = data["benchmark"]
    if benchmark == "fakeavceleb":
        clips = FakeAVCelebLoader(Path(root)).load()
    elif benchmark == "avlipsync-timit":
        clips = AVLipSyncTIMITLoader(Path(root)).load()
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    partitions = identity_disjoint_splits(
        clips, seed=seed, ratios=tuple(config.get("splits", {}).values()) or (0.7, 0.15, 0.15))
    batch_size = config["training"]["batch_size"]
    workers = config["training"].get("num_workers", 0)
    return {
        split: DataLoader(
            _make_dataset(config, split_clips, seed + offset), batch_size=batch_size,
            shuffle=split == "train", num_workers=workers, pin_memory=True,
        )
        for offset, (split, split_clips) in enumerate(partitions.items())
    }


def _batch_metrics(
    model: ContrastiveMisalignmentLearner,
    attrib: AttributionDecoder,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    attribution_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    training = optimizer is not None
    with torch.set_grad_enabled(training):
        output = model(
            tensor_batch["anchor_a"], tensor_batch["anchor_v"],
            tensor_batch["pos_a"], tensor_batch["pos_v"],
            tensor_batch["neg_a"], tensor_batch["neg_v"], tensor_batch["severity"],
        )
        _, attribution_loss = attrib(output["anomaly_direction"], tensor_batch["attribution_gt"])
        loss = output["loss"] + attribution_weight * attribution_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        "loss": float(loss.detach()),
        "contrastive_loss": float(output["loss_contrastive"]),
        "severity_loss": float(output["loss_severity"]),
        "severity_mae": float((output["severity_pred"] - tensor_batch["severity"]).abs().mean()),
        "attribution_loss": float(attribution_loss.detach()),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_real_training(config: dict[str, Any], root: str, seed: int, device: torch.device) -> dict[str, Any]:
    """Train a single seed with real, licensed clips and validation checkpointing."""
    _seed_everything(seed)
    loaders = build_real_loaders(config, root, seed)
    model_config = config["model"]
    encoder = SyncEncoder(
        embed_dim=model_config["embed_dim"], stem_embed=model_config.get("stem_embed", 128),
        num_mamba_blocks=model_config.get("num_mamba_blocks", 2),
        d_state=model_config.get("d_state", 16), keep_ratio=model_config.get("keep_ratio", 0.5),
        frame_size=model_config["frame_size"],
    ).to(device)
    model = ContrastiveMisalignmentLearner(
        encoder, embed_dim=model_config["embed_dim"], projection_dim=model_config["projection_dim"],
    ).to(device)
    attrib = AttributionDecoder(model_config["projection_dim"], num_frames=model_config["num_frames"]).to(device)
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(attrib.parameters()), lr=train_config["lr"],
        weight_decay=train_config.get("weight_decay", 0.0),
    )
    checkpoint_dir = Path("experiments/checkpoints") / config.get("campaign_name", "synctrace")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    attribution_weight = config["contrastive"].get("attribution_weight", 0.0)
    for epoch in range(train_config["epochs"]):
        model.train()
        attrib.train()
        train_metrics = _mean_metrics([
            _batch_metrics(model, attrib, batch, device, attribution_weight, optimizer)
            for batch in loaders["train"]
        ])
        model.eval()
        attrib.eval()
        val_metrics = _mean_metrics([
            _batch_metrics(model, attrib, batch, device, attribution_weight)
            for batch in loaders["val"]
        ])
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(f"METRIC epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "seed": seed, "epoch": epoch + 1, "config": config,
                "model": model.state_dict(), "attribution": attrib.state_dict(),
                "validation": val_metrics,
            }, checkpoint_dir / f"best_seed{seed}.pt")
    return {"seed": seed, "best_val_loss": best_val, "history": history,
            "n_train": len(loaders["train"].dataset), "n_val": len(loaders["val"].dataset),
            "n_test": len(loaders["test"].dataset)}


def run_smoke(args: argparse.Namespace, config: dict[str, Any], device: torch.device) -> None:
    """Small synthetic execution for CI; its output is explicitly non-benchmark."""
    model = ContrastiveMisalignmentLearner(SyncEncoder(embed_dim=args.embed_dim), embed_dim=args.embed_dim).to(device)
    attrib = AttributionDecoder(128).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(attrib.parameters()), lr=1e-3)
    for epoch in range(args.epochs):
        losses = []
        for _ in range(4):
            audio = torch.randn(2, 80, 320, device=device)
            video = torch.randn(2, 16, 3, 112, 112, device=device)
            severity = torch.tensor([0.3, 0.7], device=device)
            output = model(audio, video, audio, video, audio, video, severity)
            _, attribution_loss = attrib(output["anomaly_direction"], torch.zeros(2, 16, 7, device=device))
            loss = output["loss"] + 0.25 * attribution_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        average = float(np.mean(losses))
        print(f"METRIC mode=smoke epoch={epoch} mean_loss={average:.4f}")
    Path("experiments").mkdir(exist_ok=True)
    (Path("experiments") / f"smoke_embed{args.embed_dim}_margin{args.margin}.yaml").write_text(
        yaml.dump({"mode": "synthetic_smoke_not_benchmark", "epochs": args.epochs, "mean_loss": average}),
        encoding="utf-8",
    )
    print(f"METRIC mode=smoke final_loss={average:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--data-root", help="officially authorized benchmark root; activates real training")
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()
    config = _load_config(args.config)
    device = torch.device(args.device)
    if args.data_root:
        config["training"]["device"] = args.device
        run = run_real_training(config, args.data_root, args.seed, device)
        report = Path("experiments") / f"training_{config['data']['benchmark']}_seed{args.seed}.yaml"
        report.parent.mkdir(exist_ok=True)
        report.write_text(yaml.dump(run, sort_keys=False), encoding="utf-8")
        print(f"METRIC mode=licensed_dataset best_val_loss={run['best_val_loss']:.4f}")
    else:
        run_smoke(args, config, device)


if __name__ == "__main__":
    main()
