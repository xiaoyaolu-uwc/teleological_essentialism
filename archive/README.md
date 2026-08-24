# Archive

Code from finished phases. **Nothing here is on the live path**, but imports
have been repointed so it all still runs — an archive that does not execute is
just a diff.

The narrative for each phase is in `docs/history/`; this file says only what
the code is and what replaced it.

## `prompt_phase/`

Building and scoring the LLM labelling prompt (`d_v3`, ~91% against the 49-row
human golden set), which produced the `deploy_tag` column every later phase
treats as ground truth. Includes the OpenAI adapters, the prompt libraries, the
two-layer prompt experiment, and the evaluation harness for all of it.

Still relevant because `labelling/` imports these prompts — the labelling step
genuinely depends on this phase's output. Superseded only in the sense that the
prompt is now fixed and no longer being iterated.

## `bert_phase/`

MacBERTh fine-tuning and diagnosis: the single 4-way model, the two-stage
cascade that replaced it, and the junk-gate tooling built around it
(`ensemble_gate.py` threshold sweeps, `evaluate_text_shift.py`,
`compare_lora_sweep.py`).

**Replaced by** `models/lora/` — a Qwen3-0.6B LoRA beats MacBERTh on both dev
folds, roughly halving mix error. `train_bert.py` is kept because it documents
the cascade hypothesis this project still rests on; its shared helpers were
extracted to `models/labels.py` and `models/torch_utils.py` so the live path no
longer imports from it.

## `tests/`

Unit tests for the prompt-phase harness. They test superseded code, so they
were archived with it rather than deleted.

## `results/`

Metrics from the prompt and BERT phases. Only `metrics.json` files are tracked;
the bulk CSV output was always gitignored.
