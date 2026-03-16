"""
FastAPI server for Qwen 3.5 VLM Detection API

Endpoints:
    POST /detect         - Detect objects in a single image
    POST /detect/batch   - Detect objects in a batch of images
    GET  /health         - Health check (vLLM + API status)
    GET  /categories     - List available detection categories
"""

import base64
import logging
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from config import CATEGORIES, CONFIDENCE_THRESHOLD
from detector import QwenDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ash Qwen VLM Detection API",
    description="Qwen 3.5 based object detection for abandoned vehicles and encampments",
    version="1.0.0",
)

# Initialize detector
detector = QwenDetector()


# --- Request/Response Models ---

class DetectRequest(BaseModel):
    """Single image detection request."""
    image: str  # base64 encoded image
    categories: list[str] | None = None
    confidence_threshold: float | None = None

class BatchDetectRequest(BaseModel):
    """Batch image detection request."""
    images: list[str]  # list of base64 encoded images
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

class HealthResponse(BaseModel):
    status: str
    vllm_connected: bool


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check API and vLLM server health."""
    vllm_ok = detector.health_check()
    return HealthResponse(
        status="ok" if vllm_ok else "degraded",
        vllm_connected=vllm_ok,
    )


@app.get("/categories")
async def list_categories():
    """List available detection categories."""
    return {"categories": CATEGORIES}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    Detect objects in a single image.

    Send base64-encoded JPEG image + optional category filter.
    """
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

    # Apply custom threshold if provided
    threshold = request.confidence_threshold or CONFIDENCE_THRESHOLD
    detections = [d for d in detections if d["confidence"] >= threshold]

    elapsed_ms = (time.time() - start) * 1000

    return DetectResponse(
        detections=detections,
        inference_time_ms=round(elapsed_ms, 1),
    )


@app.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(
    file: UploadFile = File(...),
    categories: str | None = None,
):
    """
    Detect objects in an uploaded image file.

    Upload a JPEG/PNG file directly.
    """
    start = time.time()

    image_bytes = await file.read()
    cat_list = categories.split(",") if categories else CATEGORIES

    for cat in cat_list:
        if cat not in CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Unknown category: {cat}")

    detections = detector.detect_single(image_bytes, cat_list)
    elapsed_ms = (time.time() - start) * 1000

    return DetectResponse(
        detections=detections,
        inference_time_ms=round(elapsed_ms, 1),
    )


@app.post("/detect/batch", response_model=BatchDetectResponse)
async def detect_batch(request: BatchDetectRequest):
    """
    Detect objects in a batch of images.

    Send list of base64-encoded JPEG images + optional category filter.
    Recommended batch size: 8-16 images.
    """
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

    # Apply custom threshold if provided
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
