#!/usr/bin/env bash
# Waits for the corpus download to finish, then cleans every volume and builds
# the per-volume sentence CSVs. Both steps are safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "download_texts.py" > /dev/null; do sleep 30; done
echo "=== download finished: $(ls data/texts/scan_raw | wc -l) volumes ==="

echo "=== cleaning ==="
python3 corpus/clean_scan_texts.py --workers 8

echo "=== building sentences ==="
python3 corpus/build_scan_sentences.py --workers 8

echo "=== done ==="
