# LoRA Junk Gate — Per-Dimension Sweep Analysis

Snapshot analysis of the first sweep round (data pulled 2026-08-11 via
`eval/compare_lora_sweep.py`). Each section isolates one dimension by
holding everything else at the shared default (lr=2e-4, r=16/alpha=32,
target=attn, no oversample, prompt=current — this default run is
`junk_gate_lora_lr2e4` and anchors every table below).

Primary decision metric, per the project's actual objective (see
`LORA_JUNK_GATE_PLAN.md` §9): **worst-case held-out per-category recall**
(min of DT/NDT/IE) and **evenness** (max−min). A run that wins on overall
recall (`ovrl`) or junk precision (`jprec`) while sacrificing one category
is not a win — that's the exact failure mode (historically IE) this sweep
exists to catch. Golden numbers are shown for corroboration only; the
30-row golden non-junk sample is too noisy for single-run comparisons (see
Findings in `eval/lora_junk_gate_evolution.md`).

**Caveat on greedy ordering**: these runs were queued together rather than
strictly sequentially, so each comparison below uses the *original*
default (lr=2e-4, r=16, attn, no oversample, current) as its baseline —
not the previous step's winner. The recommendations at the end account for
this by flagging what still needs to be re-validated in combination.

---

## 1. Learning rate

What varied: the LoRA adapter's learning rate — 1e-4, 2e-4 (default), 3e-4.

| run | held DT/NDT/IE | even | ovrl | jprec | gold DT/NDT/IE | even | ovrl |
|---|---|---|---|---|---|---|---|
| lr1e4 | .59/.76/.56 | .21 | .70 | .83 | .30/.79/.83 | .53 | .63 |
| lr2e4 (default) | .59/.72/.52 | .20 | .66 | .82 | .50/.79/.83 | .33 | .70 |
| lr3e4 | .59/.89/.41 | .49 | .76 | .84 | .20/.71/.83 | .63 | .57 |

**Observations**: lr1e4 dominates lr2e4 on every held-out number (higher
DT/NDT/IE, comparable jprec) — strictly better, not just a different
trade-off. lr3e4 pushes NDT to .89 but craters IE to .41 and evenness to
.49, the worst evenness in the whole sweep; it wins on `ovrl` for exactly
the wrong reason (NDT is the largest true category, so overweighting it
inflates the weighted average while hiding IE collapse).

**Recommendation**: **lr=1e-4**. Clear, low-risk win.

---

## 2. LoRA rank / alpha

What varied: adapter rank/alpha — r=16/alpha=32 (default) vs r=32/alpha=64
(fixed 2:1 ratio kept constant), at lr=2e-4.

| run | held DT/NDT/IE | even | ovrl | jprec | gold DT/NDT/IE | even | ovrl |
|---|---|---|---|---|---|---|---|
| r16a32 (default) | .59/.72/.52 | .20 | .66 | .82 | .50/.79/.83 | .33 | .70 |
| r32a64 | .66/.82/.63 | .19 | .76 | .85 | .50/.71/.83 | .33 | .67 |

**Observations**: r32a64 wins on literally every column — all three
categories, evenness, overall recall, junk precision. Not a trade-off at
all at this comparison point.

**Recommendation**: **r=32/alpha=64**. Clear win, but this was only
tested against lr=2e-4, not lr=1e-4 — needs re-checking once combined.

---

## 3. Target modules

What varied: which weight matrices get LoRA adapters — attention only
(default: q/k/v/o_proj) vs attention + MLP (adds gate/up/down_proj), at
lr=2e-4, r=16.

| run | held DT/NDT/IE | even | ovrl | jprec | gold DT/NDT/IE | even | ovrl |
|---|---|---|---|---|---|---|---|
| attn (default) | .59/.72/.52 | .20 | .66 | .82 | .50/.79/.83 | .33 | .70 |
| attn_mlp | .66/.76/.59 | .17 | .72 | .82 | .30/.79/1.00 | .70 | .67 |

**Observations**: attn_mlp beats attn on all three held-out categories and
evenness, junk precision tied. Golden evenness looks much worse (.70) but
that's driven by IE=1.00 on just a handful of golden IE rows — exactly the
kind of golden-noise artifact already documented, not trustworthy at n≈30.

**Recommendation**: **target_modules=attn_mlp**, trusting held-out over
golden here.

---

## 4. Oversampling

What varied: class-weighted oversampling of non-junk rows during training
(off = default) vs on, at lr=2e-4, r=16, attn.

| run | held DT/NDT/IE | even | ovrl | jprec | gold DT/NDT/IE | even | ovrl |
|---|---|---|---|---|---|---|---|
| off (default) | .59/.72/.52 | .20 | .66 | .82 | .50/.79/.83 | .33 | .70 |
| on | .62/.72/.56 | .16 | .67 | .82 | .40/.71/.83 | .43 | .63 |

**Observations**: on improves DT and IE slightly, evenness improves
.20→.16, junk precision unchanged. Real but small effect — the smallest
delta of any dimension tested.

**Recommendation**: **oversample=True**, held with lower confidence than
the other wins given the small effect size; worth re-confirming once
combined with the other winners rather than trusting in isolation.

---

## 5. Prompt variant

What varied: the task-framing prompt prepended to the passage — none (bare
text), current (original, imprecise framing), rich (corrected
definitional/categorical framing, no examples), fewshot (1 real-quote
example pair), fewshot_multi (2 pairs), at lr=2e-4, r=16, attn, no
oversample.

| run | held DT/NDT/IE | even | ovrl | jprec | gold DT/NDT/IE | even | ovrl |
|---|---|---|---|---|---|---|---|
| none | .45/.68/.52 | .23 | .62 | .81 | .40/.71/.67 | .31 | .60 |
| current (default) | .59/.72/.52 | .20 | .66 | .82 | .50/.79/.83 | .33 | .70 |
| rich | .76/.81/.48 | .33 | .75 | .86 | .50/.71/.83 | .33 | .67 |
| fewshot | .62/.68/.59 | **.09** | .66 | .82 | .60/.71/.83 | .23 | .70 |
| fewshot_multi | .69/.86/.44 | .41 | .76 | .86 | .50/.79/.83 | .33 | .70 |

**Observations**: fewshot has the best worst-case recall (min=.59) *and*
by far the best evenness (.09 — next best is .20). rich and fewshot_multi
post higher `ovrl`/`jprec` and higher DT/NDT, but both do it by sacrificing
IE (.48 and .44 respectively) — the same mix-distorting pattern seen with
lr3e4. fewshot's golden numbers corroborate: best golden DT (.60), tied
best golden `ovrl` (.70).

**Recommendation**: **prompt=fewshot**. Note: the two `max_length=640`
truncation-check runs for fewshot/fewshot_multi are still pending
(re-submitted after hitting the SLURM time limit) — if truncation was
quietly hurting fewshot at max_length=384, the margin over fewshot_multi
could widen further once those land.

---

## Overall recommendation

Individually-tested winners: **lr=1e-4, r=32/alpha=64, target=attn_mlp,
oversample=True, prompt=fewshot**. None of these five have been tested
*together* — each comparison above holds the other four at their original
defaults, not at each other's winning values. The next concrete step is a
single combined run with all five winners applied at once, to check
whether the gains stack or partially cancel (e.g. attn_mlp adds ~2x the
trainable params of attn-only; combined with r=32 that's a much larger
adapter than any single run tested here, which could interact with lr in
ways the isolated lr sweep didn't probe).

Suggested order of confirmation once the combined run exists:
1. Compare combined run vs. each single-dimension winner — confirm no
   regression on any of held-out DT/NDT/IE/evenness.
2. Re-run rank/alpha and oversample at lr=1e-4 specifically, since both
   were only validated at lr=2e-4 and had the two smallest margins.
3. Fold in the fewshot vs. fewshot_multi max_length=640 result once
   available before locking in the prompt choice.
4. Multi-seed pass on the final combined config before trusting it as the
   answer (per the existing seed-variance finding — golden swung 0.60→0.73
   across reseeds on BERT).
