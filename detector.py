"""
Qwen 3.5 VLM detector using vLLM OpenAI-compatible API
"""

import base64
import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import (
    CATEGORY_PROMPTS,
    CONFIDENCE_THRESHOLD,
    SAMPLING_PARAMS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)

logger = logging.getLogger(__name__)


def _encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64 data URI."""
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"


def _parse_detections(text: str, category: str) -> list[dict[str, Any]]:
    """Parse JSON detections from Qwen response text."""
    # Try to find JSON array in response
    # Look for [...] pattern
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
        bbox = det.get("bbox")
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
    """Qwen 3.5 VLM detector that calls vLLM server."""

    def __init__(self, base_url: str = None, model: str = None):
        self.client = OpenAI(
            base_url=base_url or VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
        )
        self.model = model or VLLM_MODEL

    def health_check(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error("vLLM health check failed: %s", e)
            return False

    def detect_single(
        self,
        image_bytes: bytes,
        categories: list[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Run detection on a single image for specified categories.

        Args:
            image_bytes: JPEG image bytes
            categories: List of categories to detect. Defaults to all.

        Returns:
            List of detections with category, bbox, confidence, description.
        """
        if categories is None:
            categories = list(CATEGORY_PROMPTS.keys())

        all_detections = []
        image_uri = _encode_image(image_bytes)

        for category in categories:
            prompt = CATEGORY_PROMPTS.get(category)
            if not prompt:
                logger.warning("Unknown category: %s", category)
                continue

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_uri},
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    **SAMPLING_PARAMS,
                )

                text = response.choices[0].message.content
                detections = _parse_detections(text, category)
                all_detections.extend(detections)

            except Exception as e:
                logger.error("Detection failed for %s: %s", category, e)
                continue

        return all_detections

    def detect_batch(
        self,
        images: list[bytes],
        categories: list[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Run detection on a batch of images.

        Args:
            images: List of JPEG image bytes
            categories: Categories to detect. Defaults to all.

        Returns:
            List of results per image with index and detections.
        """
        results = []
        for idx, image_bytes in enumerate(images):
            detections = self.detect_single(image_bytes, categories)
            results.append({
                "image_index": idx,
                "detections": detections,
            })
        return results
