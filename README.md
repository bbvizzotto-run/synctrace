<div align="center">
  <img src="assets/synctrace_logo.png" alt="SyncTrace — A DeepFake detector" width="420"/>
</div>

# SyncTrace — Lightweight Cross-Modal Contrastive Forensics for Real-Time Audio-Visual Deepfake Detection

SyncTrace detects, localizes and explains audio-visual deepfakes by learning the
manifold of **legitimate lip-speech synchrony** exclusively from authentic
videos. No labeled synthetic deepfakes are needed: a controlled-severity
pseudo-forgery generator supplies the contrastive signal and the automatic
ground truth for localization evaluation.

## Highlights

| Contribution | What it means |
|---|---|
| Auto-supervised contrastive learning | Trained only on real videos — no dataset/generator bias from synthetic labels |
| Severity regression | Returns *how much* a video is manipulated (0–1), not just real/fake |
| Spatiotemporal attribution | Heatmaps showing which frame, facial region and audio band carry the forgery evidence |
| Edge efficiency | Mamba-Attention hybrid backbone with ONNX/TensorRT export |

## Repository layout

```
config/          YAML configs (model, training, eval, benchmark)
src/
  encoder/       SyncEncoder (Mamba-Attention hybrid)
  cml/           Contrastive Misalignment Learner + severity regressor
  sae/           Spatiotemporal Attribution Engine (heatmaps)
  data/          loaders + controlled-severity pseudo-forgery generator
  engine/        training/eval loops + benchmark harness
  app/           Streamlit/Gradio demo
  export/        ONNX/TensorRT conversion
experiments/     metric logs and comparison tables (machine-readable)
papers/          LaTeX manuscript + generated figures
.github/workflows/  CI, experiment grid runner, benchmark, paper PDF
```

## Quick start

```bash
git clone https://github.com/YOUR_USER/synctrace.git && cd synctrace
pip install -r requirements.txt
python -m src.engine.train --config config/train.yaml --epochs 50 --device cuda
python -m src.engine.benchmark --config config/benchmark.yaml
python -m src.app.run              # local demo
```

Every training run writes `experiments/run_<hparams>.yaml` containing the
config hash, seed, metrics and the exact command — full reproducibility.

## CI / automation

The repo ships GitHub Actions for lint + smoke training on every push, a
GPU experiment-grid runner (manual trigger, metrics posted to PR comments),
a benchmark job on every `v*` tag (comparison table vs. CAD / SAVe / FOMT),
and a LaTeX-to-PDF build for the manuscript.

## Citing

If you use SyncTrace, please cite the upcoming *Multimedia Tools and
Applications* paper (Springer).
