# Accuracy & Latency Report

**Status:** baseline + fine-tuning complete. CPU-latency row and demo video pending.
Full methodology and analysis: `docs/report.md` and `docs/finetuning-report.md`.

Model: `meta-llama/Llama-3.1-8B-Instruct` (open-weight, ≤ 20 B), served with vLLM 0.6.3
on a single GPU. Temperature 0 for extraction/distress; streaming on. Decoding is
unconstrained (guided/schema decoding was removed upstream to fix timeouts —
`docs/report.md` §8).

All comparisons use a **paired percentile bootstrap** (B = 5000) over the shared rows;
`code/eval_ci.py` reproduces every CI below.

---

## 1. Triage / workflow accuracy — end-to-end (59 synthetic scenarios)

`code/eval_harness.py` runs a full simulated multi-turn conversation per scenario and
scores the final disposition against a deterministically-derived gold.

| metric | baseline | 95% CI |
|---|---|---|
| triage accuracy (exact disposition) | **0.915** | [0.831, 0.983] |
| workflow accuracy (disposition + rule_id) | **0.915** | [0.831, 0.983] |
| red-flag recall — gold = `CALL_EMS_911_NOW` (17/18) | 0.944 | [0.810, 1.000] |
| under-triage rate | 0.034 | [0.000, 0.085] |
| over-triage rate | 0.051 | [0.000, 0.119] |
| conversations that did not terminate | 0 | — |
| turns / scenario (mean · p50 · p90) | 6.4 · 7 · 10 | — |

LoRA v1 end-to-end: triage 0.915, red-flag recall 0.889 — both inside the baseline CI,
i.e. **no measurable change**.

Error cost is asymmetric and classes are imbalanced, so **red-flag recall** and
**under-triage rate** are the metrics that matter, not plain accuracy. On 59 scenarios
the CI on triage accuracy is ±~8 points — sub-8-point differences are not detectable at
this eval size.

---

## 2. Latency

### 2.1 GPU (RunPod, vLLM, prefix caching on)

| per LLM call | mean | p50 | p90 |
|---|---|---|---|
| **TTFT** (s) | 0.79 | 0.32 | 2.15 |
| **total response** (s) | 3.37 | 3.38 | 6.51 |

Per-turn latency is dominated by the two LLM calls (extraction + question NLG); the
rules engine and state machine are sub-millisecond. `extract_slots` is the larger call
(carries the running conversation context). See `docs/report.md` §8 for a recurring
latency-degradation instability observed on this serving setup.

### 2.2 CPU

_[pending]_ — a ~10-scenario CPU pass (Ollama / `llama.cpp`, 8 B Q4) for real p50/p90,
or a labelled estimate (~5–20 s/turn depending on cores) with method. The CPU path is a
cost/latency comparison, not the recommended deployment.

---

## 3. Fine-tuning — baseline vs LoRA

The disposition is owned by the deterministic engine, so the LoRA is evaluated **in
isolation** on the two NLU calls it was trained on (`code/eval_extraction.py`,
`code/eval_distress.py`) — a direct measure not confounded by the rest of the pipeline.
Both runs below are on a verified-healthy server (checked via the hallucinated-key
count in the saved rows).

### 3.1 Slot extraction — 112 labelled utterances

| metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|
| row exact-match | 0.562 | 0.607 | +0.045 | [−0.036, +0.125] | no |
| key precision | 0.768 | 0.782 | +0.015 | [−0.056, +0.090] | no |
| key recall | 0.790 | 0.833 | +0.043 | [−0.022, +0.115] | no |
| key F1 | 0.779 | 0.807 | +0.028 | [−0.030, +0.098] | no |
| value accuracy | 0.844 | 0.826 | −0.018 | [−0.080, +0.044] | no |
| hallucination row-rate | 0.241 | 0.214 | −0.027 | [−0.098, +0.045] | no |

### 3.2 Distress classification — 59 utterances, 6 flags

| metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|
| micro precision | 0.636 | 0.571 | −0.065 | [−0.208, +0.069] | no |
| micro recall | 0.875 | 1.000 | +0.125 | [0.000, +0.417] | no |
| micro F1 | 0.737 | 0.727 | −0.010 | [−0.156, +0.196] | no |
| flag exact-match | 0.915 | 0.915 | 0.000 | [−0.051, +0.051] | no |

Thin-cell note: five of the six distress flags have support = 1 in this set, so per-flag
P/R is not a stable estimate; the micro-average is the figure to read.

### 3.3 Training

QLoRA, r = 16, α = 32, dropout 0.05, target `q,k,v,o` proj, 4-bit nf4, lr 2e-4,
effective batch 16, max-seq-len 2048. **v1**: 290 rows / 3 epochs / final train loss
0.66. A second iteration (**v2**: true-only distress output format + 5 epochs) regressed
the distress classifier (micro-F1 → ~0.29) and was rolled back; a third could not be
measured cleanly (serving instability, §8 of `docs/report.md`).

---

## 4. Verdict

**Every base-vs-LoRA difference has a 95% CI that includes zero** — on extraction and on
distress, isolated; and end-to-end, where v1 lands inside the baseline CI. The
fine-tune produced **no statistically detectable change** at this eval size.

This is the predicted outcome: the deterministic engine owns the safety-critical
disposition, and the LLM only extracts facts and phrases questions — tasks the base 8 B
already does near-zero-shot. **The adapter is not shipped**: it neither helps nor hurts
measurably, so it adds serving cost and a version-management burden for no benefit.

The higher-value lever, identified during this work, is a pipeline defect in how the
distress flags are merged across turns (`docs/report.md` §6.4), not fine-tuning.
