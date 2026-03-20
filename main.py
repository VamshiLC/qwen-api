"""
Qwen 3.5 VLM Detection API — single process with embedded vLLM engine.

No separate vLLM server needed. Model loads on startup, serves on port 8000.

Usage:
    python main.py
"""

import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from vllm import LLM

from config import (
    CATEGORIES,
    CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    TENSOR_PARALLEL_SIZE,
    QUANTIZATION,
    PORT,
)
from detector import QwenDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector — initialized on startup
detector: QwenDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vLLM engine on startup, cleanup on shutdown."""
    global detector
    logger.info("Loading model: %s", MODEL_NAME)
    logger.info("GPU memory utilization: %s", GPU_MEMORY_UTILIZATION)

    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        quantization=QUANTIZATION,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )

    detector = QwenDetector(engine)
    logger.info("Model loaded, API ready on port %d", PORT)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Ash Qwen VLM Detection API",
    description="Qwen 3.5 VLM detection — single process, embedded vLLM engine",
    version="2.0.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class DetectRequest(BaseModel):
    image: str
    categories: list[str] | None = None
    confidence_threshold: float | None = None

class BatchDetectRequest(BaseModel):
    images: list[str]
    categories: list[str] | None = None
    confidence_threshold: float | None = None

class Detection(BaseModel):
    category: str
    bbox: list[float]
    confidence: float
    description: str = ""

class DetectResponse(BaseModel):
    detections: list[Detection]
    inference_time_ms: float

class BatchDetectResponse(BaseModel):
    results: list[dict]
    total_images: int
    total_detections: int
    inference_time_ms: float


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen VLM Detection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }
  .container { max-width: 900px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 24px; color: #fff; }
  h1 span { color: #f90; }
  .form-row { display: flex; gap: 12px; margin-bottom: 16px; align-items: end; }
  .field { flex: 1; }
  .field label { display: block; font-size: 13px; color: #999; margin-bottom: 6px; }
  select, input[type="number"] { width: 100%; padding: 10px 12px; background: #1a1a1a; border: 1px solid #333; border-radius: 8px; color: #fff; font-size: 14px; outline: none; }
  select:focus, input:focus { border-color: #f90; }
  .upload-area { border: 2px dashed #333; border-radius: 12px; padding: 32px; text-align: center; cursor: pointer; transition: all 0.2s; margin-bottom: 16px; position: relative; }
  .upload-area:hover, .upload-area.dragover { border-color: #f90; background: #111; }
  .upload-area.has-file { border-color: #f90; border-style: solid; padding: 8px; }
  .upload-area img { max-width: 100%; max-height: 300px; border-radius: 8px; }
  .upload-text { color: #666; font-size: 14px; }
  .upload-text b { color: #f90; }
  #fileInput { display: none; }
  button { padding: 10px 24px; background: #f90; color: #000; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:hover { background: #fa0; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  .result-area { margin-top: 24px; }
  .result-img { width: 100%; border-radius: 12px; border: 1px solid #222; margin-top: 12px; }
  .stats { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
  .stat { background: #1a1a1a; border: 1px solid #222; border-radius: 8px; padding: 10px 16px; font-size: 13px; }
  .stat b { color: #f90; }
  .spinner { display: none; width: 24px; height: 24px; border: 3px solid #333; border-top-color: #f90; border-radius: 50%; animation: spin 0.7s linear infinite; margin: 0 auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: #f66; background: #1a0000; border: 1px solid #f66; border-radius: 8px; padding: 12px; margin-top: 12px; font-size: 13px; }
  .det-list { margin-top: 12px; }
  .det-item { background: #1a1a1a; border: 1px solid #222; border-radius: 8px; padding: 10px 16px; margin-bottom: 8px; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
  .det-cat { background: #f90; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }
  .canvas-wrap { position: relative; display: inline-block; margin-top: 12px; }
  canvas { max-width: 100%; border-radius: 12px; border: 1px solid #222; }
</style>
</head>
<body>
<div class="container">
  <h1><span>Qwen 3.5</span> VLM Detection</h1>

  <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
    <div class="upload-text" id="uploadText"><b>Click or drag</b> an image here</div>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <div class="form-row">
    <div class="field">
      <label>Categories</label>
      <select id="categories">
        <option value="all">All categories</option>
        <option value="abandoned_vehicle">Abandoned Vehicle</option>
        <option value="unsheltered_encampment">Unsheltered Encampment</option>
      </select>
    </div>
    <div class="field">
      <label>Confidence</label>
      <input type="number" id="threshold" value="0.2" min="0" max="1" step="0.05">
    </div>
    <button id="detectBtn" onclick="runDetect()" disabled>Detect</button>
  </div>

  <div class="result-area" id="resultArea" style="display:none">
    <div class="spinner" id="spinner"></div>
    <div class="canvas-wrap"><canvas id="resultCanvas"></canvas></div>
    <div class="stats" id="stats"></div>
    <div class="det-list" id="detList"></div>
  </div>
  <div id="errorBox"></div>
</div>

<script>
let selectedFile = null;
let imgDataUrl = null;
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');

fileInput.addEventListener('change', (e) => { if(e.target.files[0]) setFile(e.target.files[0]); });
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => { e.preventDefault(); uploadArea.classList.remove('dragover'); if(e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

function setFile(f) {
  selectedFile = f;
  detectBtn.disabled = false;
  uploadArea.classList.add('has-file');
  const reader = new FileReader();
  reader.onload = (e) => { imgDataUrl = e.target.result; uploadArea.innerHTML = '<img src="'+e.target.result+'">'; };
  reader.readAsDataURL(f);
}

const COLORS = { abandoned_vehicle: '#FF4444', unsheltered_encampment: '#FF8800' };

function drawDetections(canvas, imgSrc, detections) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    detections.forEach(d => {
      const [x1, y1, x2, y2] = d.bbox;
      const color = COLORS[d.category] || '#FF0000';
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      ctx.fillStyle = color;
      const label = d.category + ' ' + (d.confidence*100).toFixed(0) + '%';
      ctx.font = 'bold 14px sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillRect(x1, y1-20, tw+8, 20);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1+4, y1-5);
    });
  };
  img.src = imgSrc;
}

async function runDetect() {
  if(!selectedFile) return;
  const resultArea = document.getElementById('resultArea');
  const spinner = document.getElementById('spinner');
  const canvas = document.getElementById('resultCanvas');
  const stats = document.getElementById('stats');
  const detList = document.getElementById('detList');
  const errorBox = document.getElementById('errorBox');

  resultArea.style.display = 'block';
  spinner.style.display = 'block';
  canvas.style.display = 'none';
  stats.innerHTML = '';
  detList.innerHTML = '';
  errorBox.innerHTML = '';
  detectBtn.disabled = true;

  const catSelect = document.getElementById('categories').value;
  const threshold = parseFloat(document.getElementById('threshold').value);

  // Read file as base64
  const b64 = imgDataUrl.split(',')[1];
  const categories = catSelect === 'all' ? null : [catSelect];

  const body = { image: b64, confidence_threshold: threshold };
  if(categories) body.categories = categories;

  try {
    const res = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if(!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Detection failed'); }

    const data = await res.json();
    spinner.style.display = 'none';
    canvas.style.display = 'block';

    drawDetections(canvas, imgDataUrl, data.detections);

    stats.innerHTML = '<div class="stat"><b>'+data.detections.length+'</b> detections</div>'
      + '<div class="stat"><b>'+data.inference_time_ms.toFixed(0)+'</b> ms</div>';

    data.detections.forEach((d, i) => {
      detList.innerHTML += '<div class="det-item"><span><span class="det-cat">'+d.category+'</span> &nbsp; conf: <b style="color:#f90">'+d.confidence.toFixed(2)+'</b></span><span style="color:#666">'+d.description+'</span></div>';
    });
  } catch(e) {
    spinner.style.display = 'none';
    errorBox.innerHTML = '<div class="error">'+e.message+'</div>';
  }
  detectBtn.disabled = false;
}
</script>
</body>
</html>"""


@app.get("/health")
async def health():
    return {
        "status": "healthy" if detector else "loading",
        "model": MODEL_NAME,
        "engine": "vllm_embedded",
    }


@app.get("/categories")
async def list_categories():
    return {"categories": CATEGORIES}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if not detector:
        raise HTTPException(status_code=503, detail="Model still loading")

    start = time.time()

    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    categories = request.categories or CATEGORIES
    for cat in categories:
        if cat not in CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Unknown category: {cat}")

    detections = detector.detect_single(image_bytes, categories)

    threshold = request.confidence_threshold or CONFIDENCE_THRESHOLD
    detections = [d for d in detections if d["confidence"] >= threshold]

    elapsed_ms = (time.time() - start) * 1000

    return DetectResponse(
        detections=detections,
        inference_time_ms=round(elapsed_ms, 1),
    )


@app.post("/detect/batch", response_model=BatchDetectResponse)
async def detect_batch(request: BatchDetectRequest):
    if not detector:
        raise HTTPException(status_code=503, detail="Model still loading")

    start = time.time()

    if len(request.images) > 32:
        raise HTTPException(status_code=400, detail="Max 32 images per batch")

    try:
        images_bytes = [base64.b64decode(img) for img in request.images]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data in batch")

    categories = request.categories or CATEGORIES
    for cat in categories:
        if cat not in CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Unknown category: {cat}")

    results = detector.detect_batch(images_bytes, categories)

    threshold = request.confidence_threshold or CONFIDENCE_THRESHOLD
    for result in results:
        result["detections"] = [
            d for d in result["detections"] if d["confidence"] >= threshold
        ]

    total_dets = sum(len(r["detections"]) for r in results)
    elapsed_ms = (time.time() - start) * 1000

    return BatchDetectResponse(
        results=results,
        total_images=len(request.images),
        total_detections=total_dets,
        inference_time_ms=round(elapsed_ms, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
