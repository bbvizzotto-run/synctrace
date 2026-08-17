"""
SyncTrace — web demo
=====================
FastAPI app exposing a single `/analyze` endpoint plus a self-contained
HTML interface.  The demo loads the INT8-quantized ONNX graph when
`artifacts/synctrace_int8.onnx` exists (edge deployment target); otherwise
it falls back to the trained PyTorch CML model, and finally to a synthetic
(no-checkpoint) mode that still demonstrates the UI contract.

Run locally:
    pip install fastapi uvicorn
    PYTHONPATH=. python -m uvicorn src.demo.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.demo.inference import OnnxSyncTraceDemo, SyncTraceDemo

app = FastAPI(title="SyncTrace Demo", version="0.1.0")

_demo = None
_demo_info: dict = {}


def _make_demo():
    global _demo, _demo_info
    if _demo is not None:
        return
    onnx_path = Path("artifacts/synctrace_int8.onnx")
    if onnx_path.exists():
        _demo = OnnxSyncTraceDemo(onnx_path)
        _demo_info = {"backend": "onnx_int8", "model": str(onnx_path)}
        return
    try:
        import yaml

        from src.cml.contrastive_misalignment_learner import (
            ContrastiveMisalignmentLearner,
        )
        from src.encoder.sync_encoder import SyncEncoder
        from src.sae.spatiotemporal_attribution import AttributionDecoder

        with open("config/train.yaml") as fh:
            cfg = yaml.safe_load(fh)
        model_cfg = cfg.get("model", {})
        encoder = SyncEncoder(embed_dim=model_cfg.get("embed_dim", 256))
        cml = ContrastiveMisalignmentLearner(encoder,
                                             embed_dim=model_cfg.get("embed_dim", 256))
        decoder = AttributionDecoder(model_cfg.get("embed_dim", 256))
        device = "cpu"
        _demo = SyncTraceDemo(cml, decoder, device=device)
        _demo_info = {"backend": "pytorch", "device": device}
    except Exception as exc:  # noqa: BLE001
        _demo = SyncTraceDemo(None, None, "cpu")
        _demo_info = {"backend": "synthetic", "note": str(exc)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    _make_demo()
    return INDEX_HTML


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):  # noqa: B008
    _make_demo()
    if not video.filename:
        raise HTTPException(400, "no filename")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, f"unsupported extension {ext}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name
    try:
        result = _demo.score(tmp_path)
        result.update(_demo_info)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    _make_demo()
    return {"status": "ok", **_demo_info}


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SyncTrace — Audio-Video Deepfake Forensics Demo</title>
<style>
:root{--bg:#0b1020;--card:#121a33;--accent:#4f8cff;--ok:#36d399;--bad:#f87171;--txt:#e6edf7}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;min-height:100vh;padding:2rem 1rem}
.wrap{max-width:880px;margin:0 auto}
h1{font-size:1.6rem;margin-bottom:.25rem}
h1 span{color:var(--accent)}
p.sub{opacity:.7;margin-bottom:1.5rem;font-size:.9rem}
.card{background:var(--card);border:1px solid #1e2a4a;border-radius:14px;padding:1.5rem;margin-bottom:1.25rem}
label.drop{display:block;border:2px dashed #2a3a66;border-radius:12px;padding:2.5rem 1rem;text-align:center;cursor:pointer;transition:.2s}
label.drop:hover, label.drop.drag{border-color:var(--accent);background:#101a38}
input[type=file]{display:none}
button{background:var(--accent);color:#fff;border:none;border-radius:10px;padding:.75rem 1.6rem;font-size:1rem;cursor:pointer;margin-top:1rem}
button:disabled{opacity:.45;cursor:not-allowed}
.row{display:flex;gap:.75rem;flex-wrap:wrap;margin-top:1.25rem}
.stat{flex:1;min-width:120px;background:#0d1530;border-radius:10px;padding:1rem;text-align:center}
.stat .v{font-size:1.5rem;font-weight:700}
.stat .k{opacity:.6;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
.verdict-fake{color:var(--bad)}.verdict-real{color:var(--ok)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.6rem;margin-top:1rem}
.grid img{width:100%;border-radius:8px;border:1px solid #1e2a4a}
pre{white-space:pre-wrap;font-size:.75rem;opacity:.6;margin-top:.75rem;max-height:180px;overflow:auto}
#spin{display:none;margin:1rem auto;width:34px;height:34px;border:4px solid #223;border-top-color:var(--accent);border-radius:50%;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Sync<span>Trace</span></h1>
  <p class="sub">Real-time audio-video deepfake forensics via self-supervised contrastive misalignment learning.
  Upload a short video clip to get an anomaly score, a continuous manipulation severity estimate and explainable attribution heatmaps.</p>

  <div class="card">
    <label class="drop" id="drop">
      <strong>Click or drop a video here</strong><br>
      <small>MP4 / WebM / MOV / AVI — short clips (&le; 8 s) work best</small>
      <input type="file" id="file" accept="video/*">
    </label>
    <div style="text-align:center"><div id="spin"></div></div>
    <div style="text-align:center"><button id="btn" disabled>Analyze</button></div>
  </div>

  <div class="card" id="result" style="display:none">
    <div class="row">
      <div class="stat"><div class="v" id="anom">—</div><div class="k">Anomaly score</div></div>
      <div class="stat"><div class="v" id="sev">—</div><div class="k">Severity (0&ndash;1)</div></div>
      <div class="stat"><div class="v" id="vrd">—</div><div class="k">Verdict</div></div>
      <div class="stat"><div class="v" id="ms">—</div><div class="k">Inference</div></div>
    </div>
    <div id="hm-wrap" style="display:none"><h3 style="margin-top:1rem;font-size:1rem">Per-frame attribution heatmaps</h3>
      <div class="grid" id="hm"></div></div>
    <pre id="meta"></pre>
  </div>
</div>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),
      btn=document.getElementById('btn'),spin=document.getElementById('spin');
let chosen=null;
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag')});
drop.addEventListener('dragleave',()=>drop.classList.remove('drag'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files[0]){chosen=e.dataTransfer.files[0];ready()}});
file.addEventListener('change',()=>{if(file.files[0]){chosen=file.files[0];ready()}});
function ready(){drop.querySelector('strong').textContent=chosen.name;btn.disabled=false}
btn.addEventListener('click',async()=>{
  btn.disabled=true;spin.style.display='block';
  const fd=new FormData();fd.append('video',chosen);
  try{
    const r=await fetch('/analyze',{method:'POST',body:fd});
    if(!r.ok)throw new Error(await r.text());
    const j=await r.json();
    document.getElementById('result').style.display='block';
    document.getElementById('anom').textContent=j.anomaly_score;
    document.getElementById('sev').textContent=j.severity;
    const v=document.getElementById('vrd');v.textContent=j.verdict;
    v.className='v '+(j.verdict==='FAKE'?'verdict-fake':'verdict-real');
    document.getElementById('ms').textContent=(j.inference_time_ms??(j.prep_time_s*1000).toFixed(0))+' ms';
    const hm=document.getElementById('hm'),w=document.getElementById('hm-wrap');
    hm.innerHTML='';
    if(j.heatmaps_png&&j.heatmaps_png.length){w.style.display='block';
      j.heatmaps_png.forEach(s=>{const im=document.createElement('img');im.src=s;hm.appendChild(im)})}
    document.getElementById('meta').textContent='backend: '+(j.backend||'?')+' — '+((j.model&&'model: '+j.model+' — ')||'')+'prep: '+j.prep_time_s+' s';
  }catch(e){alert('Error: '+e.message)}
  finally{btn.disabled=false;spin.style.display='none'}
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
