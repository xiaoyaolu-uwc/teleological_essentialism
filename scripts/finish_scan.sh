#!/usr/bin/env bash
# Waits for the Vast scan to finish, pulls predictions home, builds the master
# sheet. Safe to re-run: the pull is an rsync and the sheet is regenerated.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== waiting for scan: $(date) ==="
while ssh -o BatchMode=yes vast 'pgrep -f "[r]un_scan_inference" > /dev/null' 2>/dev/null; do
  sleep 300
done
echo "=== scan finished: $(date) ==="
ssh vast 'tail -3 /workspace/teleo2/scan_infer.log'

echo "=== pulling predictions ==="
bash scripts/scan_on_vast.sh pull

echo "=== building master sheet ==="
python3 evaluation/build_master_sheet.py

echo "=== done: $(date) ==="
