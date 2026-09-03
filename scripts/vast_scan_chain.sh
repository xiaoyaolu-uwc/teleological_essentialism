#!/usr/bin/env bash
# Waits for deployment training to finish, verifies both adapters exist, then
# runs the cascade over the whole scanning corpus. Inference is resumable, so
# this is safe to restart.
set -uo pipefail
cd /workspace/teleo2
# Fragmentation on a 16G card is the difference between finishing and OOMing.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "train.py --run-name deploy" > /dev/null; do sleep 60; done
echo "=== training finished: $(date -u) ==="

for name in deploy_gate deploy_s2; do
  if [ ! -f "models/checkpoints/lora/$name/adapter_model.safetensors" ]; then
    echo "MISSING adapter for $name -- aborting"; exit 1
  fi
done
echo "both adapters present"

/venv/main/bin/python evaluation/run_scan_inference.py --token-budget 12000
echo "=== inference finished: $(date -u) ==="
