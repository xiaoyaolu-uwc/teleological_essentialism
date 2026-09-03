#!/usr/bin/env bash
# Ship sentences to the Vast box, run the deployment cascade there, pull the
# predictions back. Every step is resumable, so re-running is safe.
#
#   bash scripts/scan_on_vast.sh push     # sync sentences + code up
#   bash scripts/scan_on_vast.sh run      # launch inference detached
#   bash scripts/scan_on_vast.sh status   # progress
#   bash scripts/scan_on_vast.sh pull     # bring predictions home
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=vast
RDIR=/workspace/teleo2

case "${1:-}" in
  push)
    # checkpoints live on the box already; never push local ones (the obsolete
    # BERT weights alone are 1.1G and the overlay has only a few GB free)
    rsync -az --exclude '__pycache__' --exclude 'checkpoints' models/ "$REMOTE:$RDIR/models/"
    rsync -az --exclude '__pycache__' evaluation/ "$REMOTE:$RDIR/evaluation/"
    rsync -az --exclude '__pycache__' config/ "$REMOTE:$RDIR/config/"
    rsync -az data/scan/sentences/ "$REMOTE:$RDIR/data/scan/sentences/"
    ssh "$REMOTE" "du -sh $RDIR/data/scan/sentences; df -h / | tail -1"
    ;;
  run)
    ssh "$REMOTE" "cd $RDIR && setsid bash -c 'nohup /venv/main/bin/python evaluation/run_scan_inference.py --token-budget 12000 > scan_infer.log 2>&1' < /dev/null > /dev/null 2>&1 &"
    sleep 10
    ssh "$REMOTE" "ps -eo pid,etime,cmd | grep '[r]un_scan_inference' | head -1"
    ;;
  status)
    ssh "$REMOTE" "ls $RDIR/data/scan/predictions 2>/dev/null | wc -l; tail -3 $RDIR/scan_infer.log 2>/dev/null"
    ;;
  pull)
    mkdir -p data/scan/predictions
    rsync -az "$REMOTE:$RDIR/data/scan/predictions/" data/scan/predictions/
    ls data/scan/predictions | wc -l
    ;;
  *) echo "usage: $0 {push|run|status|pull}"; exit 1 ;;
esac
