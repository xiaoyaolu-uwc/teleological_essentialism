#!/usr/bin/env python3
"""Score a sample of scan predictions against the GPT-5.4 deployment labeller.

The small cascade was trained to imitate the d_v3 prompt on 16 argumentative
books. This asks whether it still does so on the scanning corpus, which is a
different genre. It reuses run_full_deployment's prompt config and parsing
unchanged, so "agreement" here means agreement with the exact labeller the
whole project is calibrated against.

Resumable: rows already scored in the output file are skipped.
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labelling.deployment_prompts import DEPLOYMENT_PROMPT_VERSIONS  # noqa: E402
import labelling.run_full_deployment as rfd  # noqa: E402

import json
import os
import random

SAMPLE = ROOT / "data/scan/validation_sample.csv"
OUT = ROOT / "data/scan/validation_scored.csv"


def parse_batch(raw, idxs):
    """Same shape as run_full_deployment's parser: a JSON list of {id, tag, ...}."""
    stripped = (raw or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return [(i, {"tag": "error", "reasoning": f"JSON parse fail: {stripped[:120]}",
                     "confidence": 0.0}) for i in idxs]
    if isinstance(parsed, dict):
        for key in ("results", "passages", "classifications"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]
    by_id = {str(r["id"]): r for r in parsed if isinstance(r, dict) and "id" in r}
    out = []
    for local, gi in enumerate(idxs):
        r = by_id.get(str(local)) or (parsed[local] if local < len(parsed) else None)
        if not isinstance(r, dict):
            out.append((gi, {"tag": "error", "reasoning": "Missing", "confidence": 0.0}))
            continue
        try:
            conf = max(0.0, min(1.0, float(r.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        out.append((gi, {"tag": str(r.get("tag", "error")).strip().lower(),
                         "reasoning": str(r.get("reasoning", ""))[:300],
                         "confidence": conf}))
    return out


async def classify_chat(client, cfg, model, texts, idxs):
    """Chat-completions equivalent of run_full_deployment.classify_batch_async.

    OpenRouter does not expose OpenAI's /responses endpoint, so the same prompt
    is sent as a system + user message pair instead. The prompt text itself is
    unchanged, which is what makes the comparison meaningful.
    """
    import openai

    block = "\n".join(cfg["passage_template"].format(id=i, text=t[:1500])
                       for i, t in enumerate(texts))
    user_msg = cfg["user_template"].format(n=len(texts), passages_block=block)

    delay = 5.0
    for attempt in range(6):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": cfg["system"]},
                          {"role": "user", "content": user_msg}],
            )
            return parse_batch(resp.choices[0].message.content, idxs)
        except openai.RateLimitError:
            if attempt == 5:
                return [(i, {"tag": "error", "reasoning": "Rate limit", "confidence": 0.0})
                        for i in idxs]
            wait = delay * (2 ** attempt)
            await asyncio.sleep(wait + random.uniform(0, wait))
        except Exception as exc:  # noqa: BLE001
            return [(i, {"tag": "error", "reasoning": str(exc)[:200], "confidence": 0.0})
                    for i in idxs]
    return [(i, {"tag": "error", "reasoning": "exhausted", "confidence": 0.0}) for i in idxs]


async def run(rows, cfg, args, client):
    sem = asyncio.Semaphore(args.concurrency)
    batches = [list(range(i, min(i + args.batch_size, len(rows))))
               for i in range(0, len(rows), args.batch_size)]

    async def bounded(idxs):
        async with sem:
            texts = [rows[i]["text"] for i in idxs]
            if args.api_style == "chat":
                return await classify_chat(client, cfg, args.model, texts, idxs)
            return await rfd.classify_batch_async(client, cfg, texts, idxs)

    results, done = {}, 0
    for coro in asyncio.as_completed([bounded(b) for b in batches]):
        for idx, r in await coro:
            results[idx] = r
        done += 1
        print(f"  batch {done}/{len(batches)}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="d_v3")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                    help="e.g. https://openrouter.ai/api/v1 for OpenRouter")
    ap.add_argument("--api-style", choices=["responses", "chat"], default=None,
                    help="defaults to chat when a base-url is set, responses otherwise")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rfd.load_dotenv()
    rfd.MODEL = args.model
    if args.api_style is None:
        args.api_style = "chat" if args.base_url else "responses"
    rows = list(csv.DictReader(SAMPLE.open()))

    already = {}
    if OUT.exists():
        for r in csv.DictReader(OUT.open()):
            # An errored row is not a scored row: retry it rather than
            # inheriting a failure from a previous run.
            if r.get("gpt_tag") and r["gpt_tag"] != "error":
                already[(r["uid"], r["sent_id"])] = r
    todo = [r for r in rows if (r["uid"], r["sent_id"]) not in already]
    # The sample file is ordered period-major, so an unshuffled --limit scores
    # a single cell and tells you nothing about the other twenty.
    random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} sampled, {len(already)} already scored, {len(todo)} to send", flush=True)
    if not todo:
        return

    import openai
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(api_key=key, base_url=args.base_url) if args.base_url \
        else openai.AsyncOpenAI(api_key=key)
    print(f"provider={'openrouter' if args.base_url else 'openai'} "
          f"style={args.api_style} model={args.model}", flush=True)

    cfg = DEPLOYMENT_PROMPT_VERSIONS[args.version]
    results = asyncio.run(run(todo, cfg, args, client))

    for i, r in enumerate(todo):
        got = results.get(i, {})
        r["gpt_tag"] = got.get("tag", "error")
        r["gpt_confidence"] = got.get("confidence", "")
        r["gpt_reasoning"] = got.get("reasoning", "")[:300]
        already[(r["uid"], r["sent_id"])] = r

    out_rows = list(already.values())
    fields = list(rows[0].keys()) + ["gpt_tag", "gpt_confidence", "gpt_reasoning"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
