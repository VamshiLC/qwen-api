#!/bin/bash
# Setup script for Qwen VLM API
# Run on the GPU instance (L40S/A10G)

set -e

echo "=== Step 1: System update ==="
sudo apt update && sudo apt install -y python3-pip python3-venv

echo "=== Step 2: Create virtual environment ==="
python3 -m venv ~/qwen-env
source ~/qwen-env/bin/activate

echo "=== Step 3: Install vLLM ==="
pip install vllm --extra-index-url https://wheels.vllm.ai/nightly

echo "=== Step 4: Install FastAPI dependencies ==="
pip install -r requirements.txt

echo "=== Step 5: Check GPU ==="
nvidia-smi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To start the API (single process):"
echo "  source ~/qwen-env/bin/activate"
echo "  python main.py"
echo ""
echo "Test:"
echo "  curl http://localhost:8000/health"
echo "============================================"
