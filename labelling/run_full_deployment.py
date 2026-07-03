#!/usr/bin/env python3
"""
run_full_deployment.py
======================
Runs the final deployment prompt over all rows in sentences.csv.
Writes results to data/sentences_labeled.csv.

Uses async concurrency (--concurrency N) to run N batches in parallel.
Default concurrency=50 is tuned for a 2M TPM limit with batch_size=5
(~5k tokens/call → ~400 calls/min at saturation → concurrency≈50).

Checkpoints after every flush — safe to interrupt and resume.
Includes exponential backoff on 429 rate-limit errors.

Usage (from repo root):
    python3 labelling/run_full_deployment.py [--concurrency 50] [--batch-size 5] [--version d_v3]
"""

import asyncio
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import PATHS
from models.deployment_prompts import DEPLOYMENT_PROMPT_VERSIONS


def load_dotenv():
    env = PATHS["env_file"]
    if env.exists():
        with open(env) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())


load_dotenv()

INPUT_CSV     = PATHS["sentences_csv"]
OUTPUT_CSV    = Path(INPUT_CSV).parent / "sentences_labeled.csv"
DEPLOY_FIELDS = ["deploy_tag", "deploy_extract", "deploy_reasoning", "deploy_confidence", "deploy_done"]


def load_input():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done_indices():
    """Return indices of rows that are successfully labeled (not errors)."""
    if not OUTPUT_CSV.exists():
        return set()
    done = set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row.get("deploy_done") == "True" and row.get("deploy_tag") != "error":
                done.add(i)
    return done


def write_output(rows, output, all_fields):
    total = len(rows)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for i in range(total):
            if i in output:
                writer.writerow(output[i])
            else:
                writer.writerow({**rows[i],
                                 "deploy_tag": "", "deploy_extract": "",
                                 "deploy_reasoning": "", "deploy_confidence": "",
                                 "deploy_done": "False"})


async def classify_batch_async(client, prompts_cfg, batch_texts, batch_indices):
    """Classify one batch asynchronously with exponential backoff on 429s."""
    import openai

    passage_tmpl = prompts_cfg["passage_template"]
    user_tmpl    = prompts_cfg["user_template"]

    passages_block = "\n".join(
        passage_tmpl.format(id=i, text=text[:1500])
        for i, text in enumerate(batch_texts)
    )
    user_msg = user_tmpl.format(n=len(batch_texts), passages_block=passages_block)

    max_retries = 6
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = await client.responses.create(
                model=MODEL,
                instructions=prompts_cfg["system"],
                input=user_msg,
            )
            break
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                print(f"  429 rate limit — giving up on batch {batch_indices[:2]}...", flush=True)
                return [(idx, {"tag": "error", "extract": "", "reasoning": "Rate limit", "confidence": 0.0})
                        for idx in batch_indices]
            base_wait = delay * (2 ** attempt)
            wait = base_wait + random.uniform(0, base_wait)  # jitter: spreads thundering herd
            print(f"  429 — retry in {wait:.1f}s (attempt {attempt+1})", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  API error on batch {batch_indices[:2]}...: {e}", flush=True)
            return [(idx, {"tag": "error", "extract": "", "reasoning": str(e), "confidence": 0.0})
                    for idx in batch_indices]

    raw = None
    for block in response.output:
        if block.type == "message":
            for part in block.content:
                if part.type == "output_text":
                    raw = part.text
                    break
        if raw:
            break

    if raw is None:
        return [(idx, {"tag": "error", "extract": "", "reasoning": "No output", "confidence": 0.0})
                for idx in batch_indices]

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return [(idx, {"tag": "error", "extract": "", "reasoning": f"JSON parse fail: {raw[:100]}", "confidence": 0.0})
                for idx in batch_indices]

    if isinstance(parsed, dict):
        for key in ("results", "passages", "classifications"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    by_id = {str(r["id"]): r for r in parsed if "id" in r}
    results = []
    for local_i, global_idx in enumerate(batch_indices):
        r = by_id.get(str(local_i)) or (parsed[local_i] if local_i < len(parsed) else None)
        if r:
            try:
                conf = float(r.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                conf = 0.5
            results.append((global_idx, {
                "tag":        r.get("tag", "error").strip().lower(),
                "extract":    r.get("extract", ""),
                "reasoning":  r.get("reasoning", ""),
                "confidence": conf,
            }))
        else:
            results.append((global_idx, {"tag": "error", "extract": "", "reasoning": "Missing", "confidence": 0.0}))
    return results


async def run_async(rows, done_indices, all_fields, batch_size, concurrency, version):
    import openai

    output = {}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                output[i] = row

    prompts_cfg = DEPLOYMENT_PROMPT_VERSIONS[version]

    pending_indices = [i for i in range(len(rows)) if i not in done_indices]
    batches = [pending_indices[s:s+batch_size] for s in range(0, len(pending_indices), batch_size)]

    total        = len(rows)
    n_done_start = len(done_indices)
    print(f"Pending batches: {len(batches)} ({len(pending_indices)} rows)", flush=True)

    client = openai.AsyncOpenAI()
    sem    = asyncio.Semaphore(concurrency)

    async def bounded(batch_indices):
        async with sem:
            texts = [rows[i]["text"] for i in batch_indices]
            return await classify_batch_async(client, prompts_cfg, texts, batch_indices)

    # Flush every concurrency batches (~1 round), so progress prints every ~250 rows
    flush_every = concurrency
    completed   = 0
    import time
    t_start = time.time()

    for chunk_start in range(0, len(batches), flush_every):
        chunk = batches[chunk_start: chunk_start + flush_every]
        tasks = [bounded(b) for b in chunk]
        results_list = await asyncio.gather(*tasks)

        for batch_results in results_list:
            for idx, pred in batch_results:
                output[idx] = {
                    **rows[idx],
                    "deploy_tag":        pred["tag"],
                    "deploy_extract":    pred["extract"],
                    "deploy_reasoning":  pred["reasoning"],
                    "deploy_confidence": pred["confidence"],
                    "deploy_done":       "True",
                }
                completed += 1

        n_labeled = n_done_start + completed
        elapsed   = time.time() - t_start
        rate      = completed / elapsed if elapsed > 0 else 0
        eta_s     = (total - n_labeled) / rate if rate > 0 else 0
        eta_min   = eta_s / 60
        print(f"  {n_labeled}/{total} ({n_labeled/total:.0%})  |  "
              f"{rate:.0f} rows/min  |  ETA ~{eta_min:.0f} min", flush=True)
        write_output(rows, output, all_fields)

    return output


MODEL = "gpt-5.4"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="gpt-5.4")
    parser.add_argument("--version",     type=str, default="d_v3",
                        help="Deployment prompt version (default: d_v3)")
    parser.add_argument("--batch-size",  type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Concurrent API calls (default: 30)")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    rows         = load_input()
    done_indices = load_done_indices()
    total        = len(rows)

    print(f"sentences.csv:  {total} rows")
    print(f"Already done:   {len(done_indices)}")
    print(f"Remaining:      {total - len(done_indices)}")
    print(f"Version:        {args.version}")
    print(f"Concurrency:    {args.concurrency}  |  Batch size: {args.batch_size}")

    if len(done_indices) == total:
        print("All rows already labeled.")
        return

    input_fields = list(rows[0].keys())
    all_fields   = input_fields + DEPLOY_FIELDS

    asyncio.run(run_async(rows, done_indices, all_fields, args.batch_size, args.concurrency, args.version))

    # Final summary
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        all_out = list(csv.DictReader(f))
    n_labeled = sum(1 for r in all_out if r.get("deploy_done") == "True")
    print(f"\nSession complete: {n_labeled}/{total} labeled.")

    if n_labeled == total:
        from collections import Counter
        dist = Counter(r["deploy_tag"] for r in all_out)
        print("Label distribution:")
        for tag, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {tag:<25} {count:>6}  ({count/total:.1%})")
        print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
