"""
SyncTrace — web demo inference pipeline
========================================
Single entry point used by the FastAPI demo app. Given an uploaded video
file, it:
  1. extracts a mel spectrogram and 16 evenly spaced frames
     (dataloaders.extract_frames / extract_audio + dataset._mel_spectrogram),
  2. runs the trained CML model (PyTorch) OR the quantized INT8 ONNX graph
     (ONNX Runtime, default) to compute anomaly score, severity and the
     128-d embedding,
  3. runs GradCAM attribution (SAE) to obtain per-frame heatmaps,
  4. returns everything as JSON-serializable structures (base64 PNG
     heatmaps + numeric scores).

This module is intentionally dependency-light on torch internals so the
demo app can also be exercised with synthetic data (no checkpoint needed).
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataloaders import extract_audio, extract_frames
from src.data.dataset import _mel_spectrogram
from src.sae.spatiotemporal_attribution import (
    SpatiotemporalAttributionEngine,
    colorize_heatmap,
)

MAX_FRAMES = 16
MAX_AUDIO_S = 8.0
SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_S)


def extract_inputs(video_path: str | Path):
    """Load a video file and return (mel (80,320), video (16,3,112,112)) tensors."""
    import torch

    video_path = Path(video_path)
    raw_audio = extract_audio(video_path, SAMPLE_RATE, MAX_AUDIO_SAMPLES)
    frames = extract_frames(video_path, MAX_FRAMES)  # (16,112,112,3) BGR
    mel = _mel_spectrogram(raw_audio, SAMPLE_RATE)[:80, :320]
    video = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(mel)).float(), \
        torch.from_numpy(np.ascontiguousarray(video)).float()


def _tensorify(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr))


def _pil_save_png(arr: np.ndarray) -> str:
    from PIL import Image

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


class SyncTraceDemo:
    """Demo inference backend (PyTorch path)."""

    def __init__(self, cml_model=None, attribution_decoder=None,
                 device: str = "cpu"):
        import torch

        self.device = torch.device(device)
        self.cml = cml_model.to(self.device) if cml_model is not None else None
        self.engine = None
        if attribution_decoder is not None and self.cml is not None:
            self.engine = SpatiotemporalAttributionEngine(attribution_decoder)
            self.engine._register_hook(self.engine.get_target_layer(self.cml))
        self._reference = None

    def set_reference_embedding(self, z: torch.Tensor) -> None:
        self._reference = z.detach().to(self.device)

    def _load_inputs(self, video_path: str | Path):
        return extract_inputs(video_path)

    def score(self, video_path: str | Path) -> dict:
        """Anomaly + severity + heatmap over one video clip."""
        import torch

        t0 = time.perf_counter()
        audio, video = self._load_inputs(video_path)
        t_prep = time.perf_counter() - t0

        audio = audio.unsqueeze(0).to(self.device)
        video = video.unsqueeze(0).to(self.device)

        if self.cml is None:
            # synthetic mode: no checkpoint available
            anomaly, severity = 0.0, 0.0
        else:
            with torch.no_grad():
                out = self.cml(audio, video, audio, video, audio, video,
                               torch.zeros(1, device=self.device))
                anomaly = float(out["anomaly_score"][0])
                severity = float(out["severity_pred"][0].clamp(0, 1))

        heatmaps = []
        if self.engine is not None:
            audio.requires_grad_(True)
            video.requires_grad_(True)
            self.cml(audio, video, audio, video, audio, video,
                     torch.zeros(1, device=self.device))
            attrib = self.engine.attribution_map(self.cml, audio, video)
            spatial = attrib["video_heatmap"][0]  # (T, H, W)
            for i in range(spatial.size(0)):
                rgb = colorize_heatmap(spatial[i]).numpy()
                heatmaps.append(_pil_save_png(rgb))

        return {
            "anomaly_score": round(anomaly, 4),
            "severity": round(severity, 4),
            "verdict": "FAKE" if anomaly > 0.5 else "REAL",
            "heatmaps_png": heatmaps,
            "n_frames": MAX_FRAMES,
            "prep_time_s": round(t_prep, 4),
        }


class OnnxSyncTraceDemo:
    """Demo inference backend using the quantized INT8 ONNX graph.

    Same contract as SyncTraceDemo but heatmap attribution is not available
    in the graph (exported graph is inference-only); the demo overlays the
    top-saliency frames returned by the audio-band energy proxy instead.
    """

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])
        self.path = str(onnx_path)

    def _load_inputs(self, video_path: str | Path):
        audio, video = extract_inputs(video_path)
        return (audio.unsqueeze(0).numpy(), video.unsqueeze(0).numpy())

    def score(self, video_path: str | Path) -> dict:
        t0 = time.perf_counter()
        a, v = self._load_inputs(video_path)
        t_prep = time.perf_counter() - t0
        t1 = time.perf_counter()
        anom, sev, _emb = self.sess.run(None, {"audio": a, "video": v})
        t_inf = time.perf_counter() - t1
        anomaly = float(anom[0])
        severity = float(np.clip(sev[0], 0.0, 1.0))
        # audio-band proxy for frame saliency (band energy weighted)
        energy = a[0].mean(axis=0)
        top = int(np.argmax(energy))
        return {
            "anomaly_score": round(anomaly, 4),
            "severity": round(severity, 4),
            "verdict": "FAKE" if anomaly > 0.5 else "REAL",
            "top_suspicious_frame": int(top),
            "inference_time_ms": round(t_inf * 1000, 2),
            "prep_time_s": round(t_prep, 4),
            "backend": "onnxruntime_int8",
            "heatmaps_png": [],
            "n_frames": MAX_FRAMES,
        }
