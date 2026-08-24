#!/usr/bin/env python3
"""
batch_size_experiment.py
========================
Tests whether batch size affects accuracy on the 49-row eval set.
Runs batch sizes [5, 10, 20, 49] × 3 runs each, with passages randomized
each run. All API calls are parallelized via asyncio.

Usage (from repo root):
    python3 eval/batch_size_experiment.py [--concurrency 20] [--model gpt-5.4]
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
from archive.prompt_phase.deployment_prompts import DEPLOYMENT_PROMPT_VERSIONS


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

PROMPT_VERSION = "d_v3"
BATCH_SIZES    = [5, 10, 20, 49]
N_RUNS         = 3


def load_eval_set():
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def classify_batch(client, sem, prompts_cfg, passages, model):
    """Single API call for a list of (local_id, text) tuples. Returns list of (local_id, tag)."""
    passage_tmpl = prompts_cfg["passage_template"]
    user_tmpl    = prompts_cfg["user_template"]

    passages_block = "\n".join(
        passage_tmpl.format(id=i, text=text[:1500])
        for i, (_, text) in enumerate(passages)
    )
    user_msg = user_tmpl.format(n=len(passages), passages_block=passages_block)

    async with sem:
        try:
            response = await client.responses.create(
                model=model,
                instructions=prompts_cfg["system"],
                input=user_msg,
            )
        except Exception as e:
            return [(orig_id, "error") for orig_id, _ in passages]

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
        return [(orig_id, "error") for orig_id, _ in passages]

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return [(orig_id, "error") for orig_id, _ in passages]

    if isinstance(parsed, dict):
        for key in ("results", "passages", "classifications"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    by_local_id = {str(r["id"]): r.get("tag", "error").strip().lower()
                   for r in parsed if "id" in r}

    results = []
    for local_i, (orig_id, _) in enumerate(passages):
        tag = by_local_id.get(str(local_i))
        if tag is None and local_i < len(parsed):
            tag = parsed[local_i].get("tag", "error").strip().lower()
        results.append((orig_id, tag or "error"))
    return results


async def run_experiment(rows, batch_size, run_idx, client, sem, prompts_cfg, model):
    """One run: shuffle rows, split into batches, call API, compute accuracy."""
    rng = random.Random(run_idx * 100 + batch_size)  # deterministic seed per (run, bs)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    # Build batches: (orig_index_in_shuffled, text)
    indexed = [(i, r["text"]) for i, r in enumerate(shuffled)]
    batches = [indexed[s:s + batch_size] for s in range(0, len(indexed), batch_size)]

    tasks = [classify_batch(client, sem, prompts_cfg, batch, model) for batch in batches]
    results_list = await asyncio.gather(*tasks)

    # Flatten results: orig_shuffled_idx → predicted_tag
    pred_map = {}
    for batch_results in results_list:
        for orig_id, tag in batch_results:
            pred_map[orig_id] = tag

    correct = 0
    errors  = 0
    for i, row in enumerate(shuffled):
        pred = pred_map.get(i, "error")
        if pred == "error":
            errors += 1
        elif pred == row["correct_tag"]:
            correct += 1

    n = len(rows)
    return {
        "batch_size": batch_size,
        "run":        run_idx + 1,
        "correct":    correct,
        "errors":     errors,
        "total":      n,
        "accuracy":   correct / n,
    }


async def main_async(model, concurrency):
    import openai

    rows       = load_eval_set()
    prompts    = DEPLOYMENT_PROMPT_VERSIONS[PROMPT_VERSION]
    client     = openai.AsyncOpenAI()
    sem        = asyncio.Semaphore(concurrency)

    # Build all 12 experiment tasks
    experiments = [
        (bs, run_i)
        for bs in BATCH_SIZES
        for run_i in range(N_RUNS)
    ]

    print(f"Running {len(experiments)} experiments "
          f"({len(BATCH_SIZES)} batch sizes × {N_RUNS} runs) "
          f"with concurrency={concurrency} ...\n")

    tasks = [
        run_experiment(rows, bs, run_i, client, sem, prompts, model)
        for bs, run_i in experiments
    ]
    results = await asyncio.gather(*tasks)

    # Sort and display
    results.sort(key=lambda r: (r["batch_size"], r["run"]))

    print(f"{'Batch':>6}  {'Run':>4}  {'Correct':>8}  {'Errors':>7}  {'Accuracy':>9}")
    print("-" * 45)

    by_bs = {}
    for r in results:
        bs = r["batch_size"]
        by_bs.setdefault(bs, []).append(r)
        err_str = f" ({r['errors']} err)" if r["errors"] else ""
        print(f"  {bs:>4}  {r['run']:>4}  {r['correct']:>5}/{r['total']}  "
              f"{'':>7}  {r['accuracy']:>8.1%}{err_str}")

    print("\n--- Summary (avg across 3 runs) ---")
    print(f"{'Batch':>6}  {'Avg accuracy':>13}  {'Min':>6}  {'Max':>6}")
    print("-" * 38)
    for bs in BATCH_SIZES:
        accs = [r["accuracy"] for r in by_bs[bs]]
        print(f"  {bs:>4}  {sum(accs)/len(accs):>12.1%}  "
              f"{min(accs):>5.1%}  {max(accs):>5.1%}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="gpt-5.4")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main_async(args.model, args.concurrency))


if __name__ == "__main__":
    main()
