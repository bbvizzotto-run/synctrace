"""
SyncTrace — SyncEncoder: hybrid Mamba-Attention backbone
==========================================================
Fuses audio and video streams into a single forensic embedding:

  - AUDIO branch:   log-mel spectrogram (F, T) -> 2D convolutional stem ->
                    Mamba-1D blocks over the time axis (linear-time
                    temporal modeling of spectral dynamics).
  - VIDEO branch:   frames (T, C, H, W) -> lightweight CNN stem per frame ->
                    sparse frame attention (attends only to high-activity
                    frames, i.e., mouth motion), pooled temporally.
  - FUSION:         cross-modal gated combination -> projection to embed_dim.

Designed to be 5-10x cheaper than full 3D-Transformer backbones used in
supervised baselines, while keeping discriminative power for cross-modal
forensic cues.
"""

import torch
from torch import nn

from src.encoder.mamba_ssm import MambaBlock


def _flops_per_second(layer_flops: float, fps: float = 25.0) -> float:
    """Helper exposed for the efficiency ablation tables."""
    return layer_flops * fps


class VisionStem(nn.Module):
    """Lightweight per-frame CNN stem (MobileNetV1-style depthwise)."""

    def __init__(self, in_channels: int = 3, embed: int = 128,
                 frame_size: int = 112):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.Conv2d(32, embed, 1, bias=False), nn.BatchNorm2d(embed),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> (B, T, embed)."""
        B, T, C, H, W = x.shape
        out = self.features(x.reshape(B * T, C, H, W))
        return out.view(B, T, -1)


class AudioStem(nn.Module):
    """Convolutional stem over the mel spectrogram -> (B, L, embed)."""

    def __init__(self, in_features: int = 80, embed: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, embed, 5, stride=2, padding=2),
            nn.BatchNorm1d(embed), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, T) -> (B, L, embed)."""
        out = self.conv(x)            # (B, embed, L)
        return out.transpose(1, 2)    # (B, L, embed)


class SparseFrameAttention(nn.Module):
    """Sparse temporal attention: keeps only the top-k highest-activity
    frames (mouth motion proxy via frame-difference energy), attends,
    and re-scales to all frames."""

    def __init__(self, embed: int, num_heads: int = 4, keep_ratio: float = 0.5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.attn = nn.MultiheadAttention(embed, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed)
        self.ffn = nn.Sequential(
            nn.Linear(embed, embed * 2), nn.GELU(), nn.Linear(embed * 2, embed))

    def _activity(self, x: torch.Tensor) -> torch.Tensor:
        """Frame activity proxy: mean abs difference vs previous frame."""
        diff = x[:, 1:].flatten(2).abs().mean(-1) - x[:, :-1].flatten(2).abs().mean(-1)
        pad = torch.zeros(x.size(0), 1, device=x.device)
        return torch.cat([pad, diff.abs()], dim=1)  # (B, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, D = x.shape
        k = max(2, int(T * self.keep_ratio))
        activity = self._activity(x)
        top_idx = activity.topk(k, dim=1).indices          # (B, k)
        gathered = torch.gather(
            x, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, k, D)
        attended, _ = self.attn(gathered, x, x)             # query=gathered
        out = self.norm(gathered + attended)
        out = out + self.ffn(out)
        # broadcast attended representation back to all frames (mean pool)
        return out.mean(dim=1)                              # (B, D)


class SyncEncoder(nn.Module):
    """Full hybrid backbone."""

    def __init__(self, embed_dim: int = 256, stem_embed: int = 128,
                 num_mamba_blocks: int = 2, d_state: int = 16,
                 keep_ratio: float = 0.5, frame_size: int = 112,
                 use_official_mamba: bool = False):
        super().__init__()
        self.audio_stem = AudioStem(in_features=80, embed=stem_embed)
        self.video_stem = VisionStem(embed=stem_embed, frame_size=frame_size)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(stem_embed, d_state=d_state,
                       use_official_kernel=use_official_mamba)
            for _ in range(num_mamba_blocks)
        ])
        self.attention = SparseFrameAttention(stem_embed, keep_ratio=keep_ratio)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.fuse = nn.Sequential(
            nn.LayerNorm(stem_embed * 2),
            nn.Linear(stem_embed * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """audio: (B, F, T) log-mel, video: (B, T, C, H, W) -> (B, embed_dim)."""
        a = self.audio_stem(audio)
        for block in self.mamba_blocks:
            a = block(a)
        a = self.audio_pool(a.transpose(1, 2)).squeeze(-1)   # (B, stem)

        v = self.video_stem(video)
        v = self.attention(v)                                # (B, stem)

        return self.fuse(torch.cat([a, v], dim=-1))

    def flops_estimate(self, frames: int = 16, mel_len: int = 320) -> float:
        """Rough FLOPs estimate for the paper's efficiency table."""
        # vision stem: depthwise convs ~ 2 * H * W * channels per frame
        v_per_frame = 2 * 112 * 112 * (16 + 16 + 32 + 32 + 256)
        a_flops = mel_len * 80 * 2 * (64 + 128 + 256)        # 3 conv layers
        mamba_flops = frames * 256 * 16 * 4 * 2              # 2 blocks
        return v_per_frame * frames + a_flops + mamba_flops
