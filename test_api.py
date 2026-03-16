"""
Quick test script for the Qwen Detection API.

Usage:
    python test_api.py <image_path>
    python test_api.py <image_path> --category abandoned_vehicle
"""

import argparse
import base64
import json
import sys
import time

import requests

API_URL = "http://localhost:8001"


def test_health():
    """Test health endpoint."""
    r = requests.get(f"{API_URL}/health")
    print(f"Health: {r.json()}")
    return r.json()["vllm_connected"]


def test_categories():
    """Test categories endpoint."""
    r = requests.get(f"{API_URL}/categories")
    print(f"Categories: {r.json()}")


def test_detect(image_path: str, categories: list[str] = None):
    """Test single image detection."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {"image": image_b64}
    if categories:
        payload["categories"] = categories

    print(f"\nDetecting in: {image_path}")
    print(f"Categories: {categories or 'all'}")

    start = time.time()
    r = requests.post(f"{API_URL}/detect", json=payload)
    elapsed = (time.time() - start) * 1000

    result = r.json()
    print(f"Status: {r.status_code}")
    print(f"Client time: {elapsed:.0f}ms")
    print(f"Server time: {result.get('inference_time_ms', 0):.0f}ms")
    print(f"Detections: {len(result.get('detections', []))}")

    for det in result.get("detections", []):
        print(f"  - {det['category']}: {det['confidence']:.2f} bbox={det['bbox']} {det.get('description', '')}")

    return result


def test_upload(image_path: str, categories: list[str] = None):
    """Test file upload detection."""
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        params = {}
        if categories:
            params["categories"] = ",".join(categories)

        r = requests.post(f"{API_URL}/detect/upload", files=files, params=params)

    result = r.json()
    print(f"\nUpload detect: {len(result.get('detections', []))} detections")
    for det in result.get("detections", []):
        print(f"  - {det['category']}: {det['confidence']:.2f}")

    return result


def test_batch(image_paths: list[str], categories: list[str] = None):
    """Test batch detection."""
    images_b64 = []
    for path in image_paths:
        with open(path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode())

    payload = {"images": images_b64}
    if categories:
        payload["categories"] = categories

    print(f"\nBatch detecting {len(image_paths)} images...")
    start = time.time()
    r = requests.post(f"{API_URL}/detect/batch", json=payload)
    elapsed = (time.time() - start) * 1000

    result = r.json()
    print(f"Status: {r.status_code}")
    print(f"Client time: {elapsed:.0f}ms")
    print(f"Total detections: {result.get('total_detections', 0)}")

    for res in result.get("results", []):
        print(f"  Image {res['image_index']}: {len(res['detections'])} detections")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen Detection API")
    parser.add_argument("image", nargs="?", help="Path to test image")
    parser.add_argument("--category", "-c", help="Specific category to detect")
    parser.add_argument("--batch", nargs="+", help="Multiple images for batch test")
    args = parser.parse_args()

    # Health check
    if not test_health():
        print("ERROR: vLLM server not connected!")
        sys.exit(1)

    test_categories()

    categories = [args.category] if args.category else None

    if args.batch:
        test_batch(args.batch, categories)
    elif args.image:
        test_detect(args.image, categories)
        test_upload(args.image, categories)
    else:
        print("\nHealth check passed. Provide an image path to test detection.")
