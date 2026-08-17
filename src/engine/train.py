"""
SyncTrace training loop — used by CI smoke runs and the GPU experiment
workflow. Prints metrics as `METRIC <name>=<value>` lines so the
experiment.yml action can scrape them for PR comments.
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.cml.contrastive_misalignment_learner import ContrastiveMisalignmentLearner
from src.sae.spatiotemporal_attribution import AttributionDecoder


class SyncEncoderStub(torch.nn.Module):
    """Stub encoder: replace with the Mamba-Attention hybrid backbone.

    Projects audio and video to the same embed_dim via separate LazyLinears
    (each learns its own input shape on first forward pass) and sums them.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.video_proj = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.LazyLinear(embed_dim),
                                              torch.nn.GELU())
        self.audio_proj = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.LazyLinear(embed_dim),
                                              torch.nn.GELU())

    def forward(self, audio, video):
        return self.video_proj(video) + self.audio_proj(audio)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--margin", type=float, default=0.3)
    args, _ = p.parse_known_args()

    torch.manual_seed(42)
    device = torch.device(args.device)

    encoder = SyncEncoderStub(embed_dim=args.embed_dim)
    model = ContrastiveMisalignmentLearner(encoder, embed_dim=args.embed_dim)
    attrib = AttributionDecoder(args.embed_dim)
    opt = torch.optim.Adam(list(model.parameters()) + list(attrib.parameters()),
                           lr=1e-3)

    # synthetic mini-batch demo (replace with dataset loaders in production)
    for epoch in range(args.epochs):
        losses = []
        for _ in range(4):
            a = torch.randn(2, 1, 16, 80, device=device)
            v = torch.randn(2, 16, 112, 112, device=device)
            sev = torch.tensor([0.3, 0.7], device=device)
            out = model(a, v, a, v, a, v, sev)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].item())

        avg = sum(losses) / len(losses)
        print(f"METRIC epoch={epoch} mean_loss={avg:.4f}")

    report = Path("experiments")
    report.mkdir(exist_ok=True)
    (report / f"run_embed{args.embed_dim}_margin{args.margin}.yaml").write_text(
        yaml.dump({"epochs": args.epochs, "mean_loss": avg,
                   "config": vars(args)}, default_flow_style=False))
    print(f"METRIC final_loss={avg:.4f}")


if __name__ == "__main__":
    main()
