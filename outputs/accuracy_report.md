# Accuracy & Latency Report

**Status:** baseline + LoRA v1 complete; LoRA v2 and the CPU-latency row are `_[pending]_`.
Full methodology and analysis: `docs/report.md` and `docs/finetuning-report.md`.

Model: `meta-llama/Llama-3.1-8B-Instruct` (open-weight, ≤ 20 B), served with vLLM on a
single GPU. Temperature 0 for extraction/distress; streaming on.

---

## 1. Triage / workflow accuracy — end-to-end (59 synthetic scenarios)

`code/eval_harness.py` runs a full simulated multi-turn conversation per scenario and
scores the final disposition against a deterministically-derived gold.

| metric | baseline | LoRA v1 | LoRA v2 |
|---|---|---|---|
| triage accuracy (exact disposition) | **0.915** | 0.915 | _[pending]_ |
| workflow accuracy (disposition + rule_id) | **0.915** | 0.915 | _[pending]_ |
| under-triage rate | 0.034 | 0.051 | _[pending]_ |
| over-triage rate | 0.051 | 0.034 | _[pending]_ |
| red-flag recall — gold = `CALL_EMS_911_NOW` (17–18) | 0.944 | 0.889 | _[pending]_ |
| red-flag recall — gold ∈ {911, ED-now} (32–33) | 0.970 | 0.939 | _[pending]_ |
| conversations that did not terminate | 0 | 0 | _[pending]_ |
| turns / scenario (mean · p50 · p90) | 6.4 · 7 · 10 | 6.6 · 7 · 11 | _[pending]_ |

Error cost is asymmetric and classes are imbalanced, so **red-flag recall** and
**under-triage rate** are the metrics that matter, not plain accuracy. Confidence
intervals: _[to add — bootstrap over the 59 scenarios]_.

---

## 2. Latency

### 2.1 GPU (RunPod, vLLM, prefix caching on)

| per LLM call | mean | p50 | p90 |
|---|---|---|---|
| **TTFT** (s) | 0.79 | 0.32 | 2.15 |
| **total response** (s) | 3.37 | 3.38 | 6.51 |

Per-turn latency is dominated by the two LLM calls (extraction + question NLG); the
rules engine and state machine are sub-millisecond. `extract_slots` is the larger call
(carries the running conversation context).

### 2.2 CPU

_[pending]_ — GPU only so far. To be added: a ~10-scenario CPU pass (Ollama /
`llama.cpp`, 8 B Q4) for real p50/p90, or a labelled estimate (~5–20 s/turn depending
on cores) with method. The CPU path is a cost/latency comparison, not the recommended
deployment.

---

## 3. Fine-tuning — the layer the LoRA actually changes

The disposition is owned by the deterministic engine, so the LoRA is evaluated **in
isolation** on the two NLU calls it was trained on. These numbers are not confounded by
the pipeline defect that flattens the end-to-end result (see `docs/report.md` §6.4).

### 3.1 Slot extraction — 112 labelled utterances (`code/eval_extraction.py`)

| metric | base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| row exact-match | 0.563 | **0.607** | _[pending]_ |
| key precision | 0.768 | 0.782 | _[pending]_ |
| key recall | 0.790 | **0.833** | _[pending]_ |
| key F1 | 0.779 | **0.807** | _[pending]_ |
| value accuracy | 0.844 | 0.826 | _[pending]_ |
| slots missed (FN) | 29 | **23** | _[pending]_ |
| slots hallucinated (FP) | 33 | 32 | _[pending]_ |

### 3.2 Distress classification — 59 utterances, 6 flags (`code/eval_distress.py`)

| metric | base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| micro precision | 0.636 | 0.571 | _[pending]_ |
| micro recall | 0.875 | **1.000** | _[pending]_ |
| micro F1 | 0.737 | 0.727 | _[pending]_ |
| flag exact-match | 0.915 | 0.915 | _[pending]_ |
| scenarios missing a needed flag | 1 | **0** | _[pending]_ |
| scenarios with a spurious flag | 4 | 5 | _[pending]_ |

Thin-cell caveat: five of the six flags have support = 1 in this set; per-flag P/R is
unstable. Micro-average and "scenarios missing a needed flag" are the trustworthy
figures.

### 3.3 Training

QLoRA, r = 16, α = 32, dropout 0.05, target `q,k,v,o` proj, 4-bit nf4, lr 2e-4,
effective batch 16, max-seq-len 2048. v1: 268 rows / 3 epochs / final loss 0.662.
v2: 290 rows / 5 epochs / final loss _[pending]_.

---

## 4. Verdict

**LoRA v1:** improves what it targets — slot-extraction key-F1 0.779 → 0.807 and
distress recall 0.875 → 1.000 (eliminates the one missed emergency) — and is neutral on
end-to-end disposition, as expected for a rules-engine architecture. It is **not shipped
on v1 alone**: the extraction value-accuracy dipped slightly and end-to-end red-flag
recall regressed (0.944 → 0.889), the latter caused by a pipeline defect
(`docs/report.md` §6.4), not the adapter. Decision after v2: ship only if it beats
baseline on extraction validity and red-flag behaviour by more than the confidence
interval.

**LoRA v2:** _[pending]_
