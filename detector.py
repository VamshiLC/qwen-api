"""
Qwen 3.5 VLM detector using embedded vLLM engine (single process).
"""

import base64
import json
import logging
import re
from typing import Any

from vllm import SamplingParams
from vllm.multimodal.utils import encode_image_base64

from config import (
    CATEGORY_PROMPTS,
    CONFIDENCE_THRESHOLD,
    SAMPLING_PARAMS as SAMPLING_CONF,
)

logger = logging.getLogger(__name__)


def _parse_detections(text: str, category: str) -> list[dict[str, Any]]:
    """Parse JSON detections from Qwen response text."""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    try:
        detections = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from response: %s", text[:200])
        return []

    if not isinstance(detections, list):
        return []

    valid = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox") or det.get("bbox_2d")
        confidence = det.get("confidence", 0.0)
        if (
            bbox
            and isinstance(bbox, list)
            and len(bbox) == 4
            and confidence >= CONFIDENCE_THRESHOLD
        ):
            valid.append({
                "category": category,
                "bbox": [float(b) for b in bbox],
                "confidence": float(confidence),
                "description": det.get("description", ""),
            })
    return valid


class QwenDetector:
    """Qwen 3.5 VLM detector with embedded vLLM engine."""

    def __init__(self, engine):
        self.engine = engine
        self.sampling_params = SamplingParams(
            temperature=SAMPLING_CONF["temperature"],
            top_p=SAMPLING_CONF["top_p"],
            max_tokens=SAMPLING_CONF["max_tokens"],
            presence_penalty=SAMPLING_CONF["presence_penalty"],
            top_k=SAMPLING_CONF["top_k"],
        )

    def detect_single(
        self,
        image_bytes: bytes,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run detection on a single image for specified categories."""
        if categories is None:
            categories = list(CATEGORY_PROMPTS.keys())

        all_detections = []
        image_b64 = base64.b64encode(image_bytes).decode()
        image_uri = f"data:image/jpeg;base64,{image_b64}"

        for category in categories:
            prompt = CATEGORY_PROMPTS.get(category)
            if not prompt:
                logger.warning("Unknown category: %s", category)
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                outputs = self.engine.chat(
                    messages=messages,
                    sampling_params=self.sampling_params,
                )

                text = outputs[0].outputs[0].text
                detections = _parse_detections(text, category)
                all_detections.extend(detections)

            except Exception as e:
                logger.error("Detection failed for %s: %s", category, e)
                continue

        return all_detections

    def detect_batch(
        self,
        images: list[bytes],
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run detection on a batch of images."""
        results = []
        for idx, image_bytes in enumerate(images):
            detections = self.detect_single(image_bytes, categories)
            results.append({
                "image_index": idx,
                "detections": detections,
            })
        return results
