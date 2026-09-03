#!/usr/bin/env bash
# Trains the two deployment models back to back on the Vast box.
# Run detached: bash scripts/vast_train_deploy.sh
set -euo pipefail
cd /workspace/teleo2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python

COMMON="--no-holdout --lora-r 32 --lora-alpha 64 --epochs 4 --text-column text \
        --max-length 640 --batch-size 4 --grad-accum-steps 4 --seed 42"

echo "=== gate: $(date) ==="
$PY models/lora/train.py --run-name deploy_gate --stage junk_gate \
    --prompt-variant A_structured $COMMON

echo "=== stage 2: $(date) ==="
$PY models/lora/train.py --run-name deploy_s2 --stage nonjunk_3way \
    --prompt-variant S2_structured $COMMON

echo "=== both done: $(date) ==="
