#!/usr/bin/env python3
"""Run the deployment cascade over the scanning corpus, one volume at a time.

Differs from run_cascade_folds.py in three ways, all because this is
deployment rather than evaluation:

  * one gate and one stage-2 model, trained on all 16 anchor works, loaded once
  * stage 2 runs only on gate survivors -- there is no true label to build the
    stage-attribution counterfactuals from, so the extra pass buys nothing
  * predictions are written to their own per-volume file, never back into the
    sentence CSV, so re-running with a retrained model cannot destroy the input

Resumable: a volume whose prediction file already exists is skipped.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.labels import STAGE_LABELS  # noqa: E402
from models.torch_utils import get_device  # noqa: E402
from models.lora.model import load_trained_model  # noqa: E402
from models.lora.train import PROMPT_VARIANTS  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

GATE_LABELS = STAGE_LABELS["junk_gate"]
SENT_DIR = ROOT / "data/scan/sentences"
PRED_DIR = ROOT / "data/scan/predictions"


def probs_for(model, tok, texts, prompt, max_length, token_budget, device):
    """Softmax probabilities for `texts`, batched shortest-first.

    Training pads every row to max_length, which is wasteful but harmless for
    12k rows. At scanning scale it is the dominant cost: the median row is
    under 60 tokens against a 640 limit. Here each batch is padded only to its
    own longest member, which is mathematically the same computation (the
    attention mask already masks padding, and the classification head reads
    the last non-pad token) for a large fraction of the time.
    """
    template = PROMPT_VARIANTS[prompt]
    # The prompt template is a fixed prefix of several hundred tokens. Ignoring
    # it made the width estimate several times too small, so "12k tokens" was
    # really 60k and the card ran out.
    prefix_tokens = 0 if template is None else len(tok(template.format(text=""))["input_ids"])

    def width_of(i: int) -> int:
        return min(max_length, prefix_tokens + len(texts[i]) // 3 + 8)

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = []
    model.eval()
    with torch.no_grad():
        start = 0
        while start < len(order):
            width = 0
            end = start
            while end < len(order):
                candidate = max(width, width_of(order[end]))
                if end > start and candidate * (end - start + 1) > token_budget:
                    break
                width = candidate
                end += 1
            idx = order[start:end]
            start = end

            # Halve on OOM rather than dying: one pathological batch must not
            # end an overnight run, and the estimate above is only an estimate.
            queue = [idx]
            while queue:
                group = queue.pop(0)
                chunk = [texts[i] for i in group]
                if template is not None:
                    chunk = [template.format(text=t) for t in chunk]
                try:
                    enc = tok(chunk, truncation=True, padding=True,
                              max_length=max_length, return_tensors="pt")
                    logits = model(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                    ).logits
                    out.append((group, F.softmax(logits.float(), dim=-1).cpu()))
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(group) == 1:
                        raise
                    half = len(group) // 2
                    queue = [group[:half], group[half:]] + queue

    n_labels = out[0][1].shape[1]
    restored = torch.empty(len(texts), n_labels)
    for group, probs in out:
        restored[torch.tensor(group)] = probs
    return restored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-ckpt", default="deploy_gate")
    ap.add_argument("--s2-ckpt", default="deploy_s2")
    ap.add_argument("--s2-stage", default="nonjunk_3way")
    ap.add_argument("--gate-prompt", default="A_structured")
    ap.add_argument("--s2-prompt", default="S2_structured")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-length", type=int, default=640)
    ap.add_argument("--token-budget", type=int, default=8000,
                    help="approx rows x padded width per batch; keeps peak memory flat")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--sent-dir", default=str(SENT_DIR))
    ap.add_argument("--pred-dir", default=str(PRED_DIR))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sent_dir = Path(args.sent_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    s2_labels = STAGE_LABELS[args.s2_stage]
    nonjunk_idx = GATE_LABELS.index("non_junk")

    ckpt_root = ROOT / "models/checkpoints/lora"
    gate, gate_tok = load_trained_model(
        args.model_name, ckpt_root / args.gate_ckpt, len(GATE_LABELS), device)
    s2, s2_tok = load_trained_model(
        args.model_name, ckpt_root / args.s2_ckpt, len(s2_labels), device)
    print(f"loaded both stages on {device}", flush=True)

    volumes = sorted(sent_dir.glob("*.csv"))
    pending = [p for p in volumes if not (pred_dir / p.name).exists()]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(volumes)} volumes, {len(pending)} pending", flush=True)

    started, rows_done = time.time(), 0
    for n, path in enumerate(pending, 1):
        rows = list(csv.DictReader(path.open()))
        if not rows:
            (pred_dir / path.name).write_text("")
            continue
        texts = [r["text"] for r in rows]

        gate_probs = probs_for(gate, gate_tok, texts, args.gate_prompt,
                               args.max_length, args.token_budget, device)
        keep_prob = gate_probs[:, nonjunk_idx]
        keep = (keep_prob >= args.gate_threshold).tolist()

        survivors = [i for i, k in enumerate(keep) if k]
        s2_probs = None
        if survivors:
            s2_probs = probs_for(s2, s2_tok, [texts[i] for i in survivors],
                                 args.s2_prompt, args.max_length, args.token_budget, device)

        out_rows = []
        for i, row in enumerate(rows):
            record = {
                "uid": row["uid"], "sent_id": row["sent_id"],
                "gate_prob_nonjunk": round(float(keep_prob[i]), 5),
                "gate_pred": "non_junk" if keep[i] else "junk",
                "s2_pred": "",
            }
            for label in s2_labels:
                record[f"s2_p_{label}"] = ""
            if keep[i]:
                j = survivors.index(i)
                probs = s2_probs[j]
                record["s2_pred"] = s2_labels[int(probs.argmax())]
                for k, label in enumerate(s2_labels):
                    record[f"s2_p_{label}"] = round(float(probs[k]), 5)
            out_rows.append(record)

        with (pred_dir / path.name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

        rows_done += len(rows)
        if n % 10 == 0 or n == len(pending):
            rate = rows_done / max(time.time() - started, 1e-6)
            print(f"[{n}/{len(pending)}] {rows_done:,} rows, {rate:.1f} rows/s", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
