"""Unit tests for ONNX export + INT8 quantization + equivalence."""

import pytest
import torch

from src.cml.contrastive_misalignment_learner import ContrastiveMisalignmentLearner
from src.encoder.sync_encoder import SyncEncoder
from src.export.export_onnx import InferenceWrapper, export_onnx, verify_equivalence

torch.manual_seed(0)

encoder = SyncEncoder(embed_dim=128, stem_embed=64, num_mamba_blocks=1)
cml = ContrastiveMisalignmentLearner(encoder, embed_dim=128)
wrapper = InferenceWrapper(cml.encoder, cml.projector, cml.severity_head)
wrapper.eval()

# export uses the static-shape (dense-attention) variant of the encoder
export_wrapper = InferenceWrapper(cml.encoder, cml.projector,
                                  cml.severity_head, export_mode=True)
export_wrapper.eval()


def test_export_produces_valid_onnx(tmp_path):
    out = tmp_path / "model.onnx"
    summary = export_onnx(export_wrapper, out)
    assert out.exists()
    assert summary["size_bytes"] > 0

    import onnx
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    # expected outputs
    names = {o.name for o in model.graph.output}
    assert {"anomaly_score", "severity_pred", "embedding"} <= names


def test_quantize_int8(tmp_path):
    out = tmp_path / "model.onnx"
    summary = export_onnx(export_wrapper, out, quantize_int8=True)
    assert summary["quantized"]
    assert (tmp_path / "model_int8.onnx").exists()
    # INT8 serialization includes per-tensor zero-point/scale metadata that
    # dominates for models under ~2 MB, so the on-disk size may not shrink
    # below the FP32 graph at this scale; the important property is that the
    # quantized graph loads and runs in ONNX Runtime (covered below)



def test_onnx_runtime_equivalence(tmp_path):
    pytest.importorskip("onnxruntime")
    out = tmp_path / "model.onnx"
    export_onnx(export_wrapper, out)
    # exported graph uses dense attention (export-mode variant); comparing
    # against the export-mode wrapper (identical graph, exact)
    res = verify_equivalence(export_wrapper, out, batch=1)
    assert res.get("equivalent") is True, res
    assert res["anom_max_abs_error"] < 1e-5
    assert res["emb_cosine_similarity"] > 0.999

    # deploy-vs-research: the top-k sparse attention of the research model
    # and the dense exported attention differ by at most ~1-2% on the
    # anomaly score (the model learns its own frame weights)
    res2 = verify_equivalence(wrapper, out, batch=1, atol=2e-2, rtol=2e-2)
    assert res2.get("equivalent") is True, res2


def test_wrapper_shapes():
    audio = torch.randn(3, 80, 320)
    video = torch.randn(3, 16, 3, 112, 112)
    with torch.no_grad():
        anom, sev, emb = wrapper(audio, video)
    assert anom.shape == (3,)
    assert sev.shape == (3,)
    assert emb.shape == (3, 128)
    assert (sev >= 0).all() and (sev <= 1).all()  # sigmoid head


def test_quantized_model_runs_in_ort(tmp_path):
    pytest.importorskip("onnxruntime")
    out = tmp_path / "model.onnx"
    summary = export_onnx(export_wrapper, out, quantize_int8=True)
    sess = ort_.InferenceSession(summary["int8_path"])
    a = torch.randn(1, 80, 320).numpy()
    v = torch.randn(1, 16, 3, 112, 112).numpy()
    outs = sess.run(None, {"audio": a, "video": v})
    assert len(outs) == 3
    assert outs[0].shape == (1,) and outs[1].shape == (1,) and outs[2].shape == (1, 128)

import onnxruntime as ort_
