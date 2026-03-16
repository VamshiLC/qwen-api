#!/bin/bash
# Setup script for ash-qwen-vllm VM
# Run this after SSH-ing into the VM

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
echo "To start vLLM server (Terminal 1):"
echo "  source ~/qwen-env/bin/activate"
echo "  vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \\"
echo "    --port 8000 \\"
echo "    --tensor-parallel-size 1 \\"
echo "    --max-model-len 4096 \\"
echo "    --quantization moe_wna16 \\"
echo "    --gpu-memory-utilization 0.95"
echo ""
echo "To start FastAPI server (Terminal 2):"
echo "  source ~/qwen-env/bin/activate"
echo "  python main.py"
echo ""
echo "Test health:"
echo "  curl http://localhost:8001/health"
echo "============================================"
