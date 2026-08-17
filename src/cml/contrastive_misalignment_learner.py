"""
SyncTrace — Contrastive Misalignment Learner (CML)
===================================================
Core scientific module. Learns the distribution of LEGITIMATE lip-speech
synchrony exclusively from authentic videos via a seeded triplet contrastive
objective over pseudo-forgeries with CONTINUOUS severity labels.

Key design:
  - Triplet (anchor, positive, negative) where anchor/positive are
    authentic-augmented views of the same clip and negative is a
    pseudo-forgery whose severity label drives a severity-aware margin.
  - A parallel severity regressor head enables quantifying HOW MUCH a video
    is manipulated (novel contribution), not just binarizing real/fake.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SeverityAwareTripletLoss(nn.Module):
    """Triplet margin loss whose margin scales with forged severity.

    Severe forgeries must be pushed further from the legitimate synchrony
    manifold than subtle ones — this yields a smooth anomaly manifold whose
    magnitude doubles as the severity score.
    """

    def __init__(self, base_margin: float = 0.3, max_margin: float = 1.2):
        super().__init__()
        self.base_margin = base_margin
        self.max_margin = max_margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        margin = self.base_margin + (self.max_margin - self.base_margin) * severity
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        return F.relu(d_pos - d_neg + margin).mean()


class SeverityRegressor(nn.Module):
    """Maps contrastive embeddings to a severity score in [0, 1]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ContrastiveMisalignmentLearner(nn.Module):
    """CML: contrastive encoder wrapper + triplet loss + severity head.

    Takes fused audio-visual embeddings from the SyncEncoder and organizes
    them into a legitimacy manifold. Downstream consumers (the classifier,
    the attribution engine) operate on distances inside this manifold.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int = 256,
                 projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder  # SyncEncoder (Mamba-Attention hybrid)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.severity_head = SeverityRegressor(projection_dim)
        self.triplet_loss = SeverityAwareTripletLoss()

    def encode(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        fused = self.encoder(audio, video)          # (B, embed_dim)
        return F.normalize(self.projector(fused), dim=-1)

    def forward(self, anchor_a, anchor_v, pos_a, pos_v, neg_a, neg_v,
                severity: torch.Tensor) -> dict:
        z_anchor = self.encode(anchor_a, anchor_v)
        z_pos = self.encode(pos_a, pos_v)
        z_neg = self.encode(neg_a, neg_v)

        loss_contrast = self.triplet_loss(z_anchor, z_pos, z_neg, severity)
        severity_pred = self.severity_head(z_neg)
        loss_severity = F.mse_loss(severity_pred, severity)

        anomaly_score = torch.stack([
            F.pairwise_distance(z_anchor[i:i + 1], z_neg[i:i + 1], p=2)
            for i in range(z_anchor.size(0))
        ])
        return {
            "loss": loss_contrast + 0.5 * loss_severity,
            "loss_contrastive": loss_contrast.detach(),
            "loss_severity": loss_severity.detach(),
            "anomaly_score": anomaly_score,
            "severity_pred": severity_pred.detach(),
        }
