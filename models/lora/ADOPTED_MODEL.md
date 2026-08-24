# Adopted junk-gate model

The final, adopted junk-gate config, recorded here so it isn't just
tribal knowledge in a chat log. See `eval/lora_prompt_evolution.md`'s
"Final decision" section for the full derivation and numbers.

## What it is

- Base model: `Qwen/Qwen3-0.6B` (public, downloads/caches automatically
  from HuggingFace Hub on first load -- not stored in this repo)
- LoRA: r=16, alpha=32, target_modules=attn (q/k/v/o_proj only -- the
  defaults; no rank increase)
- Prompt variant: `fewshot` (see `PROMPT_VARIANTS["fewshot"]` in `train.py`)
- max_length: 384
- Three independently-trained checkpoints (seeds 42, 7, 123), ensembled
  at inference (average the `non_junk` probability across all three)
- **Decision threshold: 0.70** (not the implicit 0.50 default) on the
  ensembled `non_junk` probability

Checkpoints live under `models/checkpoints/lora/`:
- `junk_gate_lora_prompt_fewshot` (seed 42)
- `junk_gate_lora_prompt_fewshot_seed7`
- `junk_gate_lora_prompt_fewshot_seed123`

These are git-ignored (binary weights, ~18MB adapter each); copy them
from wherever training happened (Marlowe: `~/teleological_essentialism/models/checkpoints/lora/`)
via `rsync`/`scp` onto any machine that needs to run inference.

## How to run it

```bash
python3 eval/ensemble_gate.py \
    --run-names junk_gate_lora_prompt_fewshot junk_gate_lora_prompt_fewshot_seed7 junk_gate_lora_prompt_fewshot_seed123 \
    --prompt-variant fewshot
```

This defaults to threshold=0.5 for the printed headline numbers; use
`--sweep` to see the full threshold curve, including 0.70. There is no
`--threshold` flag to directly select 0.70's predictions yet -- if you
need actual predicted labels at the adopted threshold (not just the
metrics table), that's a small addition to `eval/ensemble_gate.py`
(`preds_from_probs(held_ensemble, 0.70)` is already the right building
block, it's just not wired to a CLI flag).

## Verified locally

Reproduced bit-for-bit on a local Apple Silicon Mac (MPS backend,
`torch==2.6.0`, `transformers==4.57.6`, `peft==0.17.1` -- same versions
as the Marlowe training environment, just the non-CUDA build) on
2026-08-22: DT=.55, NDT=.69, IE=.59, evenness=.139, non_junk_precision=.809
at threshold=0.70, matching the numbers from Marlowe exactly.
