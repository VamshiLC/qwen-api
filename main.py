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
