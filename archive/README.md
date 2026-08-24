# Archive

Code from finished phases. **Nothing here is on the live path**, but imports
have been repointed so it all still runs — an archive that does not execute is
just a diff.

The narrative for each phase is in `docs/history/`; this file says only what
the code is and what replaced it.

## `prompt_phase/`

The harness that built and scored the LLM labelling prompt `d_v3` (~91% against
the 49-row human golden set), which produced the `deploy_tag` column every later
phase treats as ground truth. OpenAI adapters plus the evaluation scripts.

The prompts themselves are **not** here — `prompts.py` and
`deployment_prompts.py` live in `labelling/`, because that step is live and
should not import from an archive.

Deleted rather than archived: the two-layer prompt experiment (tested,
underperformed the single-pass prompt at 61.2% vs ~70%, recorded in
`docs/history/two_layer_evolution.md`) and the unused `bert.py` stub.

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

## `results/`

Kept: `final_prompt_refinement/` and `deployment/` — the iteration that produced
`d_v3`, i.e. the provenance of the labels everything else is scored against —
plus the BERT-phase metrics.

Deleted: ~15 MB of superseded prompt-iteration output (`gen4`–`gen7`,
`single_54`, `single_54mini`, `two_layer`). All of it was untracked local CSV;
the findings are preserved in the tracked logs under `docs/history/`.
