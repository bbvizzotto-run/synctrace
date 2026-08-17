"""
SyncTrace — Selective State Space Model block (Mamba-style)
=============================================================
A self-contained Mamba-style SSM block: linear time processing of sequences
with input-dependent (selective) dynamics.

Two implementations:
  1. If `mamba_ssm` (or `mamba-ssm`) is installed, uses the official kernel.
  2. Otherwise, a pure-PyTorch reference implementation (discretized S4-style
     selective SSM) that keeps the same API and semantics for research
     reproducibility — this is what the MTAP article's ablations rely on.

Usage:
    block = MambaBlock(d_model=256, d_state=16, d_conv=4, expand=2)
    out = block(x)   # x: (B, L, D) -> (B, L, D)
"""

import torch
import torch.nn.functional as F
from torch import nn

try:
    from mamba_ssm import Mamba as OfficialMamba  # type: ignore
    _HAS_MAMBA = True
except ImportError:  # pragma: no cover - tested without official kernel
    OfficialMamba = None
    _HAS_MAMBA = False


def _has_official_mamba() -> bool:
    return _HAS_MAMBA


class SelectiveSSMCell(nn.Module):
    """Discretized selective SSM cell (reference implementation).

    Dynamics: h_t = A(h_{t-1}) + B(u_t) * u_t,  y_t = C * h_t
    A, B depend on the input (selection mechanism); discretized via ZOH with
    dt predicted from the input.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # input-dependent parameters
        self.dt_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model * d_state, bias=False)
        self.c_proj = nn.Linear(d_model, d_model * d_state, bias=False)
        # A is input-independent but scaled by dt
        self.a_log = nn.Parameter(torch.empty(d_model * d_state))
        nn.init.uniform_(self.a_log, -2.0, -0.5)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, D) -> y: (B, L, D)."""
        B, L, D = u.shape
        dt = F.softplus(self.dt_proj(u))                    # (B, L, D)
        b = self.b_proj(u).view(B, L, D, self.d_state)      # (B, L, D, N)
        c = self.c_proj(u).view(B, L, D, self.d_state)
        a = -torch.exp(self.a_log).view(D, self.d_state)    # (D, N) stable

        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)                 # (B, D, 1)
            b_t = b[:, t]                                    # (B, D, N)
            u_t = u[:, t, :].unsqueeze(-1)                   # (B, D, 1)
            a_bar = torch.exp(a.unsqueeze(0) * dt_t)         # (B, D, N)
            h = a_bar * h + (b_t * dt_t) * u_t               # selective update
            ys.append((c[:, t] * h).sum(-1))                 # (B, D)
        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """Residual Mamba block: Norm -> SSM(expanded) -> projection -> residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, use_official_kernel: bool = True):
        super().__init__()
        self.use_official = use_official_kernel and _HAS_MAMBA
        d_inner = d_model * expand

        if self.use_official:
            self.ssm = OfficialMamba(d_model=d_inner, d_state=d_state)
        else:
            self.ssm = SelectiveSSMCell(d_inner, d_state)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        if not self.use_official:
            # causal conv1d pre-processing (reference impl)
            self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        if not self.use_official:
            x = x.transpose(1, 2)
            x = self.conv(x)[..., : x.shape[-1]]
            x = F.silu(x).transpose(1, 2)
        else:  # pragma: no cover - official kernel path
            x = F.silu(x)
        x = self.ssm(x)
        return self.out_proj(x) + residual
