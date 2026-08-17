"""Tests for the SyncTrace web demo (FastAPI app + inference pipeline)."""

import pytest

pytest.importorskip("fastapi")


@pytest.fixture()
def tmp_video(tmp_path):
    """Tiny synthetic AV file via ffmpeg (or a fallback raw MP4-like stub)."""
    import subprocess

    out = tmp_path / "clip.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=112x112:rate=25",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "libx264", "-c:a", "aac", "-t", "2",
                str(out),
            ],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("ffmpeg not available")
    return out


def test_pipeline_onnx_score(tmp_video):
    from src.demo.inference import OnnxSyncTraceDemo

    demo = OnnxSyncTraceDemo("artifacts/synctrace_int8.onnx")
    res = demo.score(tmp_video)
    assert isinstance(res["anomaly_score"], float)
    assert 0 <= res["severity"] <= 1
    assert res["verdict"] in {"REAL", "FAKE"}
    assert res["n_frames"] == 16
    assert res["backend"] == "onnxruntime_int8"


def test_pipeline_pytorch_score(tmp_video):

    from src.cml.contrastive_misalignment_learner import (
        ContrastiveMisalignmentLearner,
    )
    from src.demo.inference import SyncTraceDemo
    from src.encoder.sync_encoder import SyncEncoder
    from src.sae.spatiotemporal_attribution import AttributionDecoder

    cml = ContrastiveMisalignmentLearner(SyncEncoder(embed_dim=256))
    demo = SyncTraceDemo(cml, AttributionDecoder(256))
    res = demo.score(tmp_video)
    assert isinstance(res["anomaly_score"], float)
    assert res["verdict"] in {"REAL", "FAKE"}
    # 16 GradCAM heatmaps rendered as PNG data-URIs
    assert len(res["heatmaps_png"]) == 16
    assert res["heatmaps_png"][0].startswith("data:image/png;base64,")


def test_app_analyze(tmp_video):
    from fastapi.testclient import TestClient

    from src.demo.app import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        with open(tmp_video, "rb") as fh:
            r = client.post("/analyze",
                            files={"video": ("clip.mp4", fh, "video/mp4")})
        assert r.status_code == 200, r.text
        j = r.json()
        assert "anomaly_score" in j and "severity" in j and "verdict" in j


def test_app_rejects_bad_extension(tmp_video):
    from fastapi.testclient import TestClient

    from src.demo.app import app

    with TestClient(app) as client:
        with open(tmp_video, "rb") as fh:
            r = client.post("/analyze",
                            files={"video": ("bad.txt", fh, "text/plain")})
        assert r.status_code == 400


def test_index_html():
    from fastapi.testclient import TestClient

    from src.demo.app import app

    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "SyncTrace" in r.text
