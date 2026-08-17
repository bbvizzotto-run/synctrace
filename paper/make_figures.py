"""Generate figures for the SyncTrace manuscript (matplotlib, no external deps)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.edgecolor": "#333",
})

BLUE = "#2b5ea7"
ORANGE = "#d97706"
GREEN = "#15803d"
PURPLE = "#7e22ce"
GRAY = "#6b7280"


def box(ax, x, y, w, h, label, color, fs=8.5, sub=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                       facecolor=color, edgecolor="#111", linewidth=0.8, alpha=0.9)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), label,
            ha="center", va="center", fontsize=fs, color="white", weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.10, sub, ha="center", va="center",
                fontsize=6.2, color="white", alpha=0.92)


def arrow(ax, x1, y1, x2, y2, color="#111", lw=1.1):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, color=color)
    ax.add_patch(a)


# ----------------------------------------------------------------------------
# Figure: pipeline overview
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 4.6)
ax.axis("off")

# --- A: pseudo-forgery generator ---
box(ax, 0.2, 1.5, 2.2, 1.6, "(a) Pseudo-forgery\ngenerator", GRAY)
ax.text(1.3, 2.55, "s in [0,1] severity", ha="center", fontsize=6.2, color="#333")
ax.text(1.3, 1.75, "frames T-mask\npixels S-mask\nmel band-mask", ha="center",
        fontsize=6.2, color="#333")
arrow(ax, 2.4, 2.3, 3.0, 2.3)

# --- B: training batch ---
box(ax, 3.0, 3.0, 1.9, 0.85, "anchor +\npos. augment", BLUE)
box(ax, 3.0, 1.5, 1.9, 0.85, "neg. pseudo-\nforgery, sev. s", ORANGE)
arrow(ax, 4.9, 3.4, 5.7, 3.0, color=BLUE)
arrow(ax, 4.9, 1.9, 5.7, 2.6, color=ORANGE)

# --- C: SyncEncoder ---
box(ax, 5.7, 2.0, 2.4, 2.0, "(b) SyncEncoder", BLUE, sub="mel-stem · Mamba SSM\nvision-stem · sparse-attn")
arrow(ax, 8.1, 3.0, 8.7, 3.0)

# --- D: CML ---
box(ax, 8.7, 2.0, 2.6, 2.0, "(c) CML", PURPLE,
    sub="contrastive manifold\ntriplet + severity margin\nregressor s̃ ∈ [0,1]")
arrow(ax, 10.0, 2.0, 10.0, 1.35, color=PURPLE)

# --- outputs ---
box(ax, 8.7, 0.3, 1.25, 0.85, "anomaly\nA(z)", ORANGE)
box(ax, 10.05, 0.3, 1.25, 0.85, "severity\ns̃", ORANGE)
arrow(ax, 10.0, 1.25, 10.0, 1.17, color=PURPLE)

# --- E: SAE (below) ---
box(ax, 0.2, 0.05, 7.5, 1.0,
    "(d) SAE: per-frame GradCAM, lips / lower-face / full-face, 4 mel bands, mIoU / P@k / AUROC",
    GREEN, fs=8)
arrow(ax, 7.7, 2.05, 5.0, 1.05, color=GREEN, lw=1.0)
arrow(ax, 3.0, 1.05, 2.5, 0.05, color=GREEN, lw=1.0)

fig.tight_layout()
fig.savefig("fig_pipeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_pipeline.png saved")
