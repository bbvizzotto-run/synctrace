"""Unit tests for the SyncEncoder hybrid backbone."""

import torch

from src.encoder.mamba_ssm import SelectiveSSMCell
from src.encoder.sync_encoder import SyncEncoder, _flops_per_second


def test_selective_ssm_cell_shape_and_time():
    cell = SelectiveSSMCell(d_model=32, d_state=8)
    x = torch.randn(2, 16, 32)
    y = cell(x)
    assert y.shape == (2, 16, 32)


def test_sync_encoder_shapes():
    enc = SyncEncoder(embed_dim=256, stem_embed=64, num_mamba_blocks=1)
    enc.eval()
    audio = torch.randn(2, 80, 320)
    video = torch.randn(2, 16, 3, 112, 112)
    out = enc(audio, video)
    assert out.shape == (2, 256)


def test_sync_encoder_backward():
    enc = SyncEncoder(embed_dim=64, stem_embed=32, num_mamba_blocks=1)
    audio = torch.randn(1, 80, 320)
    video = torch.randn(1, 8, 3, 112, 112)
    out = enc(audio, video)
    out.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_flops_helper():
    assert _flops_per_second(1e6, fps=25.0) == 25e6
