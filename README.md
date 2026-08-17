<div align="center">
  <img src="assets/synctrace_logo.png" alt="SyncTrace — A DeepFake detector" width="420"/>
</div>

# SyncTrace

**Real-time audio–video deepfake detection via self-supervised contrastive misalignment learning and localizable spatiotemporal attribution.**

Manuscript in preparation for *Multimedia Tools and Applications* (Springer).

---

SyncTrace detects, localizes and explains audio–visual lip-sync forgeries by
learning the manifold of **legitimate lip–speech synchrony** exclusively from
authentic videos. No labeled synthetic deepfakes are required: a
controlled-severity pseudo-forgery generator supplies both the contrastive
signal and the automatic ground truth for localization evaluation.

## Scientific contributions

| Contribution | What it means |
|---|---|
| Self-supervised contrastive learning | Trained only on real videos — no dataset/generator bias from synthetic labels |
| Continuous severity regression | Returns *how much* a video is manipulated (0–1), not just real/fake |
| Localizable explainability | GradCAM-based heatmaps over frames, facial regions (lips / lower face / full face) and mel-frequency audio bands, with objective metrics (mIoU, Precision@k, AUROC) |
| Edge-efficient backbone | Hybrid Mamba SSM + sparse frame Attention (SyncEncoder, ~1.30 M parameters), exportable to ONNX opset 17 with INT8 dynamic quantization — ≈14 ms per clip on a commodity CPU |

## Repository layout

```
config/                       Declarative YAML configuration (train.yaml)
paper/                        Snapshot PDF do manuscrito (fonte LaTeX no Overleaf/Prism)
src/
  encoder/                    SyncEncoder — Mamba–Attention hybrid backbone
  cml/                        Contrastive Misalignment Learner + severity regressor
  sae/                        Spatiotemporal Attribution Engine (heatmaps + mIoU metrics)
  data/                       Loaders (FakeAVCeleb, AV-LipSync-TIMIT), PyTorch dataset
                              and controlled-severity pseudo-forgery generator
                              (MediaPipe FaceMesh, 478 landmarks, automatic GT masks)
  engine/                     Training loop and benchmark harness (METRIC output)
  demo/                       FastAPI web app: video upload → anomaly score,
                              severity, verdict and heatmap visualization
  export/                     ONNX export (torch dynamo, opset 17) + INT8 quantization
                              and exact PyTorch↔ONNX Runtime equivalence verification
tests/                        32 unit/integration tests (pytest)
experiments/                  Reproducible run configs and benchmark results
.github/workflows/            ci.yml (lint + smoke), experiment.yml (GPU grid + PR comments)
scripts/check_run.sh          Helper to poll GitHub Actions run status
```

## Quick start

```bash
git clone https://github.com/bbvizzotto-run/synctrace.git && cd synctrace
pip install -r requirements.txt

python -m src.engine.train --config config/train.yaml --epochs 50 --device cuda
python -m src.engine.benchmark --config config/benchmark.yaml
python -m src.export.export_onnx --out artifacts/synctrace.onnx --quantize --verify

# Web demo (FastAPI, uses the INT8 ONNX graph at ~14 ms/clip on CPU)
python -m uvicorn src.demo.app:app --port 8000   # open http://localhost:8000
```

Run the test suite with `PYTHONPATH=. pytest tests/` (32 tests) and lint with
`ruff check src tests`.

Every training/benchmark run emits `METRIC`-formatted lines that the CI
workflows parse and post to pull-request comments, and writes a
reproducible `experiments/run_<hparams>.yaml` snapshot with the config hash,
seed, metrics and the exact command.

## CI / automation

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push | `ruff` lint + full pytest suite + 1-epoch smoke training |
| `experiment.yml` | Manual dispatch | Auto-detects GPU (falls back to CPU), runs the benchmark grid, posts AUC/EER/AP, severity MAE, mIoU and efficiency results as a PR comment |

## Citing

If you use SyncTrace, please cite the upcoming *Multimedia Tools and
Applications* paper (Springer).
