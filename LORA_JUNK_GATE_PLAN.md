# Plan: LoRA-Fine-Tuned LLM Junk Gate

Status: **draft for discussion — nothing in this plan has been built yet.**

This doc has two jobs: (1) teach the general shape of a LoRA fine-tuning
project, since this is your first one, and (2) lay out the specific decisions
and options for *this* task so we can align before writing code. Read it,
push back on anything, and we'll turn the agreed-on version into an execution
checklist.

---

## 1. Where this fits

Recap from `eval/bert_cascade_evolution.md` and project memory: the cascade
(junk gate → 3-way DT/NDT/IE classifier) works. Stage 2 is strong (golden
acc 0.87–0.90). Stage 1 — the junk/non-junk gate — is the bottleneck, and its
failure mode is specific: it disproportionately throws away real
`internal_essence` content as junk. Two BERT variants (MacBERTh,
bert-base-uncased) were both diagnosed on this and neither fixed it — it's
not a "wrong pretrained weights" problem.

The hypothesis for this session: a decoder-style LLM, even a small one, has
more contextual/semantic capacity than BERT-base-class encoders, and might
draw the junk/non-junk line more like the GPT-5.4 labeler did (91.8% on the
golden set) than like MacBERTh did. LoRA lets us test that without the cost
of full fine-tuning.

Stage 2 (`nonjunk_3way`) is *not* being touched. Whatever we build here needs
to slot into `eval/evaluate_cascade.py`'s two-stage interface so it can be
swapped in for the existing `junk_gate_loo_reign_of_law` checkpoint and
compared directly against the measured BERT baseline on Reign of Law +
golden set (§9).

---

## 2. The general workflow (what an ML engineer actually does here)

Skipping "what is a neural net" since you know the theory — this is the
practical sequence, in order, with why each step exists:

1. **Frame the task as a model interface.** Before touching a model you
   decide: what goes in, what comes out, and what object computes the loss.
   For us: input = a sentence-length historical text extract; output = a
   binary label. The open question is *how* the LLM produces that label —
   see §3, this is the single biggest fork in the whole plan.

2. **Pick a base model.** Small enough to iterate on quickly, big enough to
   plausibly beat BERT at this specific discrimination. Constrained by what
   hardware you'll actually train on (§4).

3. **Prep data in the format the chosen interface needs.** We already have
   the labels (`sentences_train.csv`, `deploy_tag`/`deploy_extract`) — this
   step is about *reshaping*, not relabeling: e.g. wrapping each row in a
   prompt template if we go generative, or just tokenizing raw text if we
   go classification-head. This is mechanical once §3 is decided.

4. **Attach LoRA adapters.** Instead of updating all of the base model's
   weights (slow, memory-heavy, and risks wrecking the model's general
   language ability), LoRA freezes the base model and injects small
   trainable low-rank matrices into specific layers (typically the
   attention projections). You train only those — usually <1% of total
   params. This is *why* fine-tuning an LLM on a laptop or a single GPU is
   feasible at all.

5. **Train.** Loop over the data, compute loss, backprop into the adapter
   weights only, checkpoint the adapter (not the whole model — that's the
   other big win of LoRA, checkpoints are tens of MB, not gigabytes).

6. **Evaluate on held-out data the same way the BERT gate was evaluated.**
   This is non-negotiable for the comparison to mean anything: same LOO
   texts, same golden set, same metrics (`metrics_from_preds` in
   `models/train_bert.py` — accuracy, macro F1, per-class recall/precision,
   confusion matrix).

7. **Wire it into the cascade** and re-run `eval/evaluate_cascade.py`-style
   end-to-end scoring with the LoRA gate as stage 1 and the existing BERT
   `nonjunk_3way` checkpoint as stage 2, unchanged.

8. **Write up findings** the way `eval/bert_cascade_evolution.md` does —
   what worked, what didn't, and why, so a future session doesn't re-litigate
   settled questions.

---

## 3. The central decision: classification head vs. generative

This is the fork that shapes everything downstream, so it's worth its own
section rather than burying it in an options list.

**Option A — Classification head (BERT-style, LLM backbone).**
Take the base LLM, strip the language-modeling head, attach a small linear
classification head on top (2 outputs: junk / non-junk), LoRA-tune the
backbone plus train that head. This is exactly what
`AutoModelForSequenceClassification` does — same API you already used for
MacBERTh in `train_bert.py`, just pointed at a decoder model instead of an
encoder. Loss is plain cross-entropy over 2 classes. Eval is
`argmax(logits)`, directly reusing `evaluate()` / `metrics_from_preds()`
from `train_bert.py` almost unchanged.

**Option B — Generative / instruction-tuned (the "actually prompt an LLM"
approach).**
Format each row as an instruction ("Classify this passage as junk or
non_junk: {text}") and fine-tune the model to generate the label as text
("junk" / "non_junk"). This is closer to how LoRA fine-tuning is usually
described in tutorials and how you'd fine-tune a model meant for chat/
instruction use. It requires: a prompt template, generation-based inference
(not argmax — you generate tokens and parse the output), and care around
things like making sure the model doesn't ramble past the label.

**Tradeoffs:**

| | A: classification head | B: generative |
|---|---|---|
| Conceptual novelty vs. BERT work | Low — same shape, new backbone | High — new eval machinery, new failure modes (malformed output, verbosity) |
| Eval comparability to BERT gate | Direct, same code | Needs a parsing/normalization layer first |
| How "LLM fine-tuning" is usually taught/done in practice | Less typical | More typical — this is what most LoRA tutorials mean |
| Risk of scope creep | Lower | Higher (prompt engineering creeps back in) |
| Pedagogical value for *you* specifically | Teaches LoRA mechanics cleanly, isolated from prompting concerns | Teaches LoRA + prompt templating + generation-based eval — more surface area |

**Decided: Option A.**

Clarification from discussion: A vs. B is not "prompt or no prompt" — both
options can and should still wrap the raw sentence in a task-framing prompt
before tokenizing (e.g. `"Classify this passage from a historical
natural-history text: {text}"`), since Qwen is instruction-tuned and likely
responds better to a framed input than to bare text (which is all BERT ever
got). The actual fork is what that input feeds into: in A, a frozen-shape
classification head (linear layer → 2-way softmax, `argmax` at eval time,
same as BERT). In B, the language-modeling head itself, trained to
*generate* the label as text, which requires generation-based inference and
output parsing. We're doing A: prompt the input, classify with a head, skip
the generation/parsing machinery.

---

## 4. Model choice

You have Marlowe (H100s) and Farmshare (L40S) available, which changes the
calculus from "what fits on a MacBook's MPS" to "what's worth the extra
training time for this specific task." A binary junk/non-junk decision is
not a task that obviously benefits from a 7B+ model — the ceiling is set by
whether the model can tell "this sentence asserts an explanatory claim about
an animal trait" from "this sentence doesn't," not by broad world knowledge.

Reasonable candidates, smallest to largest:
- **Qwen2.5-0.5B / 1.5B** — fast, cheap, good LoRA target, probably enough
  signal to know if the *approach* works before spending a bigger budget.
- **Llama-3.2-1B/3B** or **Qwen2.5-3B** — meaningfully more capacity, still
  trains in minutes-to-an-hour per epoch on an H100/L40S for a dataset this
  size (~40k rows, short sequences).
- **Llama-3.1-8B / Qwen2.5-7B** — full-size territory. With H100 access this
  is feasible, but probably overkill for a binary gate, and iteration speed
  (the thing that actually matters for a learning-focused session) drops.

Suggested approach: start at the small end, get the full pipeline working
end-to-end and see where it lands relative to BERT's baseline (§5 table),
*then* decide if scaling up is worth it. Cheap to fail fast small; expensive
to debug a broken pipeline big.

**Decided model: Qwen3-0.6B to start, Qwen3-1.7B as the step-up if needed.**

Checked current state of the art (Aug 2026) rather than assuming: Qwen3
(dense, 0.6B/1.7B/4B/8B/...) supersedes Qwen2.5 at the same sizes — Qwen3's
1.7B/4B/8B outperform larger Qwen2.5 models on over half of benchmarks, and
0.6B/1.7B are well-supported in the `transformers`/`peft` ecosystem with
32K context. Other options surveyed: Llama-3.2-1B (fine, but Qwen3 benchmarks
better and has no gated-download friction), Gemma 4 e2b (multimodal-focused,
no benefit for a text-only task), SmolLM2-1.7B (aimed at CPU/phone
deployment, not a training-quality argument in our favor). Qwen3-0.6B is the
closest analog to Qwen2.5-0.5B I originally floated, but newer and stronger
at the same budget.

Sources: [Qwen3 Full Model Lineup Guide 2026](https://baeseokjae.github.io/posts/qwen-3-full-lineup-guide-2026/), [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1), [Best Small LLM for Local Deployment in 2026](https://www.ertas.ai/best/best-small-llm-for-local-deployment)

---

## 5. Training framework: manual loop vs. HF `Trainer`/`peft`

`train_bert.py` uses a hand-rolled training loop (explicit `for epoch...for
batch`, manual optimizer step, manual checkpointing-on-best-val-F1). That
was a deliberate, transparent choice — good for understanding exactly what's
happening.

For LoRA, the standard tool is Hugging Face's `peft` library
(`get_peft_model`, `LoraConfig`) layered on top of either:
- the same manual loop style (just wrap the model in `peft` first, everything
  else about the loop is unchanged), or
- HF's `Trainer` class, which handles the loop, checkpointing, and logging
  for you but hides more of the mechanics.

Given you explicitly want to *see* how this is built rather than call a
black box, I'd suggest: **manual loop, `peft`-wrapped model** — mirrors
`train_bert.py` closely enough that the diff between "fine-tune BERT" and
"LoRA-fine-tune an LLM" is almost entirely the `peft` wrapping step, which
is the actual new concept. This is my default; flag if you'd rather use
`Trainer` (less code to write, more idiomatic HF, but more magic).

---

## 6. Repo organization — options

Three ways to slot this into the existing structure:

**Option 1 — Mirror the BERT layout exactly.**
```
models/train_lora.py          # parallel to train_bert.py
models/lora.py                # parallel to bert.py (ModelAdapter stub)
models/checkpoints/lora_junk_gate_*/   # alongside existing BERT checkpoints
eval/results/lora/
eval/lora_junk_gate_evolution.md      # parallel to bert_cascade_evolution.md
```
Pro: zero new structure to learn, directly discoverable next to the BERT
code it's compared against. Con: `models/` becomes a mix of encoder and
decoder fine-tuning code with similar-looking filenames.

**Option 2 — Standalone subsystem.**
```
llm_finetune/
    train.py
    data.py            # duplicates or imports the CSV-loading logic
    requirements.txt   # scoped deps (peft, accelerate, bitsandbytes...)
    checkpoints/
eval/lora_junk_gate_evolution.md
```
Pro: clean separation — this is a genuinely different modeling paradigm, and
it may end up with cluster-specific tooling (SLURM scripts, etc.) that has
no business in `models/`. Con: some duplication of the data-loading/metrics
code already in `train_bert.py` unless we deliberately factor it out.

**Option 3 — Middle ground: shared utilities extracted first.**
Pull the parts of `train_bert.py` that aren't BERT-specific — `load_train_pool`,
`load_golden_eval`, `stratified_val_split`, `metrics_from_preds` — into a
new `models/data_utils.py`, then add:
```
models/lora_gate/
    train.py
    model.py
models/data_utils.py   # shared by train_bert.py and lora_gate/train.py
```
Pro: avoids duplicating the data pipeline (which is the part most likely to
silently drift out of sync if copy-pasted), namespaces the new paradigm
cleanly. Con: touches `train_bert.py` (low-risk refactor, but it's a working
file) before writing any new code.

**Decided: Option 3.** The data-loading/eval code must stay identical
between BERT and LoRA runs for the comparison to be valid — factoring it out
once removes the risk of it drifting. Concretely: extract `load_train_pool`,
`load_golden_eval`, `stratified_val_split`, `metrics_from_preds` out of
`models/train_bert.py` into `models/data_utils.py`, then add
`models/lora_gate/train.py` + `models/lora_gate/model.py` alongside it.
`train_bert.py` itself keeps working, just imports from the new module
instead of defining those functions locally — this refactor is chunk 1 of
the execution checklist, before any new LoRA code is written.

Noted for later, not now: the repo overall needs a cleanup pass (too much
mixed together — `eval/` alone holds evolution docs, analysis scripts, and
results side by side). Worth a dedicated session once the LoRA gate work
lands, not folded into this plan.

---

## 7. Compute/workflow logistics

**Decided: standard flow — I write the training script plus a SLURM
submission script, you `rsync`/`git push` to Marlowe or Farmshare and launch
it yourself, then pull the results back down.** This is the standard
academic-cluster pattern and worth learning properly since you'll use it
beyond this project. Because LoRA adapters are small (tens of MB, not the
full model), round-tripping results is cheap — you're not shipping gigabytes
back and forth.

I'll teach this as its own execution chunk before we touch training code:
SSH access, syncing the repo (`rsync` vs. `git clone`+`git pull`, and when
each is more appropriate), what a SLURM job script actually contains
(`#SBATCH` directives — partition, GPU count/type, time limit, output log
path), submitting (`sbatch`), monitoring (`squeue`, `sacct`, tailing the log
file), and pulling results back (`rsync` the checkpoint + `metrics.json`
directories down). Marlowe and Farmshare may differ in partition
names/module systems — I'll ask you for the specifics (or have you paste in
their docs) when we get to that chunk rather than guessing.

---

## 8. Repo cleanup (deferred)

Flagged in §6: the repo is a real mess and needs a dedicated pass — but
that's out of scope for *this* plan. Tracked here so it isn't lost.

---

## 9. Evaluation protocol and target numbers

**Why not a full LOO sweep, and why not a fresh text:** a full leave-one-out
sweep across many texts means training many models per config — expensive,
and mostly useful for *diagnosing* generalization variance, not what we need
here. A fresh, untested text (Bridgewater VII Vol. 1 was considered — best
tag balance in the corpus) was rejected on reflection: without a BERT
cascade baseline *on that exact text*, any "target" we set for it would be
a number borrowed from a different holdout and passed off as the bar to
clear, which isn't a valid comparison.

**Decided holdout: Reign of Law** — already has a real BERT cascade run
(`eval/results/bert_cascade/loo_reign_of_law/metrics.json`), so the
baseline below is measured on the exact same held-out rows the LoRA gate
will be evaluated on, not approximated from a different text. (Darwiniana
was the other option with an existing run, but its DT/IE counts, 11 and 22
rows, are thin enough that its baseline numbers are noisy — e.g. one config
below quietly means "100% of 3.")

**Baseline, measured directly** (pulled from the metrics.json above, verified
against `nonjunk3way_loo_reign_of_law/metrics.json` for the isolated-stage-2
numbers):

| Metric | Held-out text (n=469) | Golden set (n=49) |
|---|---|---|
| Stage-1 (gate) accuracy | 0.751 | 0.673 |
| Stage-1 non_junk recall | 0.467 | 0.600 |
| End-to-end accuracy / macro-F1 | 0.706 / 0.482 | 0.653 / 0.650 |
| End-to-end recall — divine_teleology | 31.0% (9/29) | 40.0% (4/10) |
| End-to-end recall — non_divine_teleology | 38.1% (43/113) | 64.3% (9/14) |
| End-to-end recall — internal_essence | 22.2% (6/27) | 66.7% (4/6) |

(bert-base-uncased, evaluated on this same holdout, reached golden non_junk
recall 0.80 and golden acc 0.776 — the best single number seen on Reign of
Law with any BERT variant; worth citing as the toughest baseline to beat,
not just MacBERTh's.)

**Targets** — beat both the MacBERTh numbers above and bert-base's 0.80/0.776
best case:

| Metric | Baseline (numbers to beat) | LoRA target |
|---|---|---|
| Held-out non_junk recall | 0.467 | **≥ 0.65** |
| Golden non_junk recall | 0.600 (0.800 for bert-base) | **≥ 0.85** |
| Held-out end-to-end recall, each of DT/NDT/IE | 22.2–38.1% | **≥ 55%** on each |
| Golden end-to-end macro-F1 | 0.650 | **≥ 0.75** |
| Golden end-to-end DT recall | 40.0% | **≥ 0.70**, approaching stage-2-alone's isolated ceiling (0.90 golden, per `nonjunk3way_loo_reign_of_law`) |

Chosen to require closing most of the gap to stage-2's own isolated ceiling
(0.775 held-out / 0.900 golden, from running stage 2 directly on all true
non-junk rows, gate bypassed) — that's the real ceiling for how good an
end-to-end cascade *could* get if the gate stopped being the bottleneck.

North star, not a hard target: GPT-5.4's 91.8% golden accuracy on the full
labeling task. Not expected to reach that with a 0.6B–1.7B model — included
only so "how far from the ceiling are we" stays visible.

### Update: pooled non_junk recall superseded by per-category proportion metrics

The targets above (§9, original) were framed around pooled non_junk recall
and end-to-end cascade recall per category. Revised framing, decided later:
the project's actual deliverable is accurate category *proportions* across
a text, which pooled recall can look fine on while still being systematically
wrong — e.g. if the gate keeps NDT at a much higher rate than IE, the pooled
average hides that IE's estimated share of the corpus comes out too low.

Three metrics now anchor evaluation instead (see `models/data_utils.py`'s
`gate_proportion_metrics`, and `eval/compare_lora_sweep.py`):
1. **Per-category recall** (DT/NDT/IE individually) through the gate alone —
   no cascade run needed, since a row's true category is known regardless of
   what a downstream classifier would guess.
2. **Recall evenness** (max−min across the three) — the direct measure of
   the distortion-risk described above.
3. **Precision** (junk leakage into non_junk) — already captured by the
   existing `junk` per-class precision in `holdout_text_metrics`, no new
   computation needed.

Additionally, per-category recall lets us compute the actual category *mix*
without running stage 2 at all: assume a perfect downstream 3-way classifier
(a survivor is credited to its true category), and compare the resulting
mix against the text's true non-junk composition. This isolates gate-induced
distortion from stage-2 error entirely.

**BERT baseline, recomputed under this framing** (Reign of Law, from the
already-stored `eval/results/bert_cascade/loo_reign_of_law/metrics.json` —
no rerun needed, since "predicted ≠ junk" already means "gate said
non_junk" regardless of what stage 2 guessed):

| Metric | Held-out text (n=469) | Golden set (n=49) |
|---|---|---|
| Per-category recall — DT | 0.483 | 0.400 |
| Per-category recall — NDT | 0.496 | 0.714 |
| Per-category recall — IE | 0.333 | 0.667 |
| Recall evenness (max−min) | **0.162** | 0.314 |
| True non-junk mix (DT/NDT/IE) | 17.2% / 66.9% / 16.0% | 33.3% / 46.7% / 20.0% |
| Survivor mix, perfect-stage-2 assumption | 17.7% / 70.9% / **11.4%** | 22.2% / 55.6% / 22.2% |

IE's share shrinks from 16.0%→11.4% among held-out survivors — the
essentialism-specific distortion from project memory, now quantified
directly rather than inferred from the false-junk-rate analysis.

**Revised targets** (held-out, since golden's n=30 non-junk rows makes
evenness especially noisy there — one flipped row moves it a lot):
- Per-category recall ≥ 0.65 on **each** of DT/NDT/IE (not just the pooled
  average clearing that bar)
- Recall evenness ≤ 0.10 (down from BERT's 0.162)
- Junk precision (no regression on leakage) ≥ BERT's held-out junk
  precision of 0.752

`eval/compare_lora_sweep.py` now sorts runs by worst-case per-category
recall on held-out, not pooled non_junk recall, so a run can't win by being
strong on two categories while abandoning the third.

---

## 10. Decisions — resolved

1. **Classification head vs. generative** (§3): **A**, with a task-framing
   prompt still wrapping the input.
2. **Starting model** (§4): **Qwen3-0.6B**, step up to Qwen3-1.7B if needed.
3. **Repo organization** (§6): **Option 3** — extract shared utils to
   `models/data_utils.py`, new code in `models/lora_gate/`.
4. **Cluster workflow** (§7): **standard flow** — script + SLURM, you run it
   on Marlowe/Farmshare, taught step by step.
5. **Evaluation protocol** (§9): **single holdout — Reign of Law** +
   golden set, chosen because a real BERT cascade baseline already exists
   for it (measured, not estimated), target numbers set above.

Next: turn §2's eight steps into a concrete, numbered execution checklist —
one chunk per step, built and explained one at a time. First chunk is the
`data_utils.py` extraction (§6), second is the cluster access walkthrough
(§7), before any LoRA-specific code gets written.
