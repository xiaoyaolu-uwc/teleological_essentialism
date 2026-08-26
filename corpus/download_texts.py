#!/usr/bin/env python3
"""Download OCR text for every volume in the scanning pool.

Reads the two manifests (BHL scan pool, IA 1750-1799 supplement), fetches each
volume's OCR text, and records the outcome. Resumable: a volume whose text file
already exists on disk is skipped, and the status log is rewritten each pass,
so the job can be killed and restarted at any point.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import csv
import random
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
BHL_POOL = ROOT / "corpus/bhl/derived/scan_pool.csv"
IA_POOL = ROOT / "corpus/ia/derived/ia_supplement_candidates.csv"
OUT_DIR = ROOT / "data/texts/scan_raw"
STATUS = ROOT / "data/texts/scan_download_status.csv"

CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "Mozilla/5.0 (Stanford teleology corpus research; contact marcus)"}
MIN_BYTES = 5_000


def targets() -> list[dict]:
    out = []
    for row in csv.DictReader(BHL_POOL.open()):
        out.append({
            "source": "BHL", "uid": f"bhl_{row['item_id']}", "year": row["year"],
            "work_id": row["work_id"], "title": row["title"][:150],
            # BHL's own itemtext endpoint returns 403; every BHL item is
            # mirrored on Internet Archive under its barcode.
            "url": f"https://archive.org/download/{row['barcode']}/{row['barcode']}_djvu.txt",
        })
    for row in csv.DictReader(IA_POOL.open()):
        out.append({
            "source": "IA", "uid": f"ia_{row['identifier']}", "year": row["year"],
            "work_id": row["work_key"], "title": row["title"][:150],
            "url": row["item_text_url"],
        })
    return out


def fetch(url: str, attempts: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, context=CTX, timeout=180) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - network errors are all retryable here
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404):
                raise
            time.sleep(2 ** attempt + random.random())
    raise last  # type: ignore[misc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between requests")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = all)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = targets()
    print(f"{len(rows)} volumes in pool", flush=True)

    pending, results, skipped = [], [], 0
    for row in rows:
        path = OUT_DIR / f"{row['uid']}.txt"
        if path.exists() and path.stat().st_size >= MIN_BYTES:
            skipped += 1
            results.append({**row, "status": "cached", "bytes": path.stat().st_size, "error": ""})
        else:
            pending.append(row)
    if args.limit:
        pending = pending[: args.limit]
    print(f"{skipped} cached, {len(pending)} to fetch", flush=True)

    counters = {"ok": 0, "failed": 0}
    lock = threading.Lock()

    def work(row: dict) -> dict:
        path = OUT_DIR / f"{row['uid']}.txt"
        try:
            data = fetch(row["url"])
            if len(data) < MIN_BYTES:
                outcome = {**row, "status": "too_short", "bytes": len(data), "error": ""}
            else:
                path.write_bytes(data)
                outcome = {**row, "status": "ok", "bytes": len(data), "error": ""}
        except Exception as exc:  # noqa: BLE001
            outcome = {**row, "status": "error", "bytes": 0, "error": str(exc)[:120]}
        with lock:
            counters["ok" if outcome["status"] == "ok" else "failed"] += 1
            total = counters["ok"] + counters["failed"]
            if total % 50 == 0:
                print(f"[{total}/{len(pending)}] ok={counters['ok']} failed={counters['failed']}", flush=True)
        time.sleep(args.delay)
        return outcome

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results.extend(pool.map(work, pending))

    downloaded, failed = counters["ok"], counters["failed"]

    with STATUS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDONE downloaded={downloaded} cached={skipped} failed={failed}", flush=True)
    print(f"status -> {STATUS}", flush=True)


if __name__ == "__main__":
    main()
