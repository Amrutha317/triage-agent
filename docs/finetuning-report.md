# Fine-tuning the chest-pain triage NLU layer with QLoRA

**Status:** draft — v1 results complete, v2 (5-epoch retrain) pending.
**Branch:** `sft-dataset-and-model-fix` · **PR:** #1
**Base model:** `meta-llama/Llama-3.1-8B-Instruct`, served with vLLM 0.6.3 on a single GPU.

---

## 1. Summary

The triage agent is a deterministic rules engine (`rules.yaml` + `decision_engine.py` +
`state_machine.py`) wrapped around an 8B LLM that does two narrow jobs — **slot
extraction** and **distress classification**. The LLM never picks a disposition.

Baseline end-to-end accuracy is **91.5%** (54/59 scenarios). All five failures are in
the LLM layer; the rules engine is correct on all five in offline mode.

A QLoRA adapter was trained on a 290-row set built specifically against those failures,
and **evaluated in isolation on the two NLU calls** rather than only end-to-end (an
end-to-end number runs through a separate, identified pipeline defect that masks the
adapter's effect).

**v1 result (measured, isolated):**

| layer | metric | base | LoRA v1 | Δ |
|---|---|---|---|---|
| extraction (112 rows) | row exact-match | 0.563 | **0.607** | +0.045 |
| | key-F1 | 0.779 | **0.807** | +0.028 |
| | key recall | 0.790 | **0.833** | +0.043 |
| | slots missed | 29 | **23** | −6 |
| | value accuracy | 0.844 | 0.826 | −0.018 |
| distress (59 rows) | micro recall | 0.875 | **1.000** | +0.125 |
| | scenarios missing a needed flag | 1 | **0** | −1 |
| | micro precision | 0.636 | 0.571 | −0.065 |
| | micro F1 | 0.737 | 0.727 | −0.010 |

The adapter **trades precision for recall** — the correct direction for triage — and
eliminates the one genuinely-missed emergency at baseline. End-to-end accuracy is
unchanged because of the pipeline defect in §7.

**v2 result:** _[pending — 5-epoch retrain on 290 rows with true-only distress flags +
25 value-accuracy rows]_

Three latent bugs were fixed along the way (§8): a train/serve model mismatch, a
training-sequence truncation, and a GPU-placement failure.

---

## 2. System under test

```
patient text ──┬─► llm_client.classify_distress()  ─► observed red-flag / triager slots
               └─► llm_client.extract_slots()       ─► {slot: value}
                        │
                        ▼
               state_machine.ingest(merged slots)  ─► Turn (ask | final)
                        │
                        ▼
               decision_engine  ─► first rules.yaml rule that fires  ─► Disposition
```

- **`extract_slots`** — converts a patient utterance into a schema-constrained
  `{slot: value}` dict. System prompt `EXTRACT_SYS` (~1,640 tokens), guided-JSON decoding.
- **`classify_distress`** — reads *how* the patient presents and emits six observed-only
  red-flag / triager-judgment booleans. System prompt `DISTRESS_SYS`, deliberately
  calibrated to over-flag ("a missed emergency costs a life").
- Everything downstream is deterministic. `slots.py` is the single schema authority.

---

## 3. Baseline evaluation

### 3.1 End-to-end (`code/eval_harness.py`, `data/eval/scenarios.jsonl`, 59 scenarios)

| metric | value |
|---|---|
| triage accuracy (exact disposition) | 0.915 |
| workflow accuracy (disposition + rule_id) | 0.915 |
| under-triage rate | 0.034 |
| over-triage rate | 0.051 |
| red-flag recall (gold = 911) | 0.944 |
| red-flag recall (gold ∈ {911, ED-now}) | 0.970 |
| did not terminate | 0 |
| turns / scenario | mean 6.4, p50 7, p90 10 |
| call TTFT (s) | p50 0.32, p90 2.15 |
| call total (s) | p50 3.38, p90 6.51 |

### 3.2 The five failures — all LLM-layer

| scenario | gold → predicted | root cause | call |
|---|---|---|---|
| `ems: triager judges life-threatening` | `CALL_EMS_911` → `HOME_CARE` | distress classifier under-called `life_threatening` on the concatenated chief complaint | distress |
| `educ: triager judges very sick/weak` | `GO_TO_ED_UCC` → `HOME_CARE` | flag set on turn 1, then cleared downstream (§7) | distress |
| `ed: difficulty breathing (non-severe)` | `GO_TO_ED_NOW` → `CALL_EMS_911` | classifier promoted mild "short of breath" to `severe_difficulty_breathing` | distress |
| `educ: pleuritic pain (worse on deep breath)` | `GO_TO_ED_UCC` → `CALL_EMS_911` | classifier read "hurts to breathe in" as a breathing emergency | distress |
| `pcp3: nitro-resolved` | `SEE_PCP_3_DAYS` → `GO_TO_ED_NOW` | extractor hallucinated `other_symptoms=[difficulty_breathing]` from the vague answer "some of those" | extract |

Four distress calibration errors (two under-, two over-), one extractor over-extraction.

---

## 4. Approach

**Fine-tune the two NLU calls; measure them directly.**

End-to-end scenario accuracy only reflects extraction/distress quality *indirectly*, and
it runs through the rest of the pipeline — including the defect in §7. So the primary
metric is two isolated evals built for this work:

- **`code/eval_extraction.py`** — runs the 112-row `data/eval/extraction_golden.jsonl`
  (one hand-labeled utterance → exact slot dict per row) through `extract_slots()`.
  Reports row exact-match, key-level precision/recall/F1 (an FP key = a hallucinated
  slot — the `pcp3` failure mode), value accuracy (enum-sets as sets, floats with
  tolerance), a per-slot table, and every hallucination row.
- **`code/eval_distress.py`** — derives a labeled set from the 59 scenarios' `facts`
  (each chief-complaint utterance + its six observed-flag values), runs
  `classify_distress()` once per row, reports per-flag precision/recall/F1 and which
  scenarios miss a needed flag or raise a spurious one.

Neither touches the state machine or rules engine, so the number reflects the model.

---

## 5. Datasets

Three distinct artifacts — easy to conflate:

| file | rows | one row is | role |
|---|---|---|---|
| `data/eval/scenarios.jsonl` | 59 | a full patient case → correct disposition | end-to-end eval |
| `data/eval/extraction_golden.jsonl` | 112 | one utterance → exact slot dict | isolated NLU eval |
| `data/train/sft_v1.jsonl` | 290 | one LLM call + gold JSON, chat form | **training** |

### 5.1 How the training set is built (`code/generate_sft_set.py`)

Deterministic, no network, no LLM. Two prompt families:

- **extract** — system = `EXTRACT_SYS` imported verbatim from `llm_client.py`; user
  mirrors `LLMClient.extract_slots()` exactly (known-slots line, "asked about" line,
  the triple-quoted message). Response = the gold slot JSON.
- **distress** — system = `DISTRESS_SYS` verbatim; user mirrors
  `LLMClient.classify_distress()`. Response = the gold flag JSON.

Because the prompts are byte-identical to serving, the adapter transfers. The canonical
schema is `slots.py`; every extraction label is passed through
`llm_client.validate_slots()` at build time and the build aborts on any invalid label.
`tests/test_sft_set.py` re-checks shape, verbatim prompts, strict-JSON responses, and
schema-valid labels.

The set is **independent** of `extraction_golden.jsonl` (different wording), so the
isolated eval measures generalisation, not memorisation.

### 5.2 Composition (v2, 290 rows: 219 extract / 71 distress)

Weighted toward the five failures. Notable buckets:

| bucket | rows | targets |
|---|---|---|
| `distress_life_threatening` / `distress_very_sick` | 14 / 12 | failures 1, 2 (under-call) |
| `distress_mild_dyspnea_negative` / `distress_pleuritic_negative` | 6 / 5 | failures 3, 4 (over-call) |
| `distress_calm_negative` | 14 | anti-escalation counterweight |
| `anti_overextract` / `underspecified` / `no_new_info` / `correction` | 10 / 8 / 3 / 2 | failure 5 + general precision |
| `bystander_not_patient` / `stent_not_angina` / `onset_precision` / `central_location_not_extracted` | 4 / 5 / 6 / 3 | v2: v1's value-accuracy misses |
| full slot / enum coverage (duration, pattern, radiation, nitro, PE, GERD, …) | rest | |

### 5.3 v1 → v2 changes (driven by measured v1 results)

1. **Distress rows emit the four optional flags only when `true`** (v1 emitted all six
   booleans every row). v1's format taught the adapter to assert
   `visible_facial_diaphoresis: false` etc. on calm turns; the base model omits them.
   `life_threatening` / `very_sick_or_weak` / `rationale` are still always present.
2. **+25 extraction rows** for v1's value-accuracy regressions:
   - `bystander_not_patient` — "my dad had a heart attack at 50" is
     `cardiac_risk_factors=[strong_family_history]`, **not** `age=50` +
     `history_of_heart_disease` (v1 pulled the bystander's age).
   - `stent_not_angina` — a stent / bypass / prior MI is `history_of_heart_disease`;
     it is **not** a diagnosis of angina (`known_angina_history`).
   - `onset_precision` — v1 got `onset_hours_ago` as a key but the wrong number.
   - `central_location_not_extracted` — "middle of my chest" stated in passing is not
     extracted as `location`.

---

## 6. Training setup (`code/finetune_lora.py`)

| | value |
|---|---|
| method | QLoRA (4-bit nf4, double quant, bf16 compute) |
| LoRA | r=16, α=32, dropout 0.05, target `q_proj,k_proj,v_proj,o_proj` |
| trainable params | 13.6 M / 8.04 B (0.17%) |
| optimizer / schedule | paged_adamw_8bit, cosine, warmup 0.03, lr 2e-4 |
| batch | 2 × grad-accum 8 (effective 16) |
| max-seq-len | 2048 (extraction rows peak at ~1,800 tokens; 0/290 truncated) |
| gradient checkpointing | on |
| **v1** | 268 rows, 3 epochs, ~50 steps, final train loss 0.662 |
| **v2** | 290 rows, 5 epochs, ~90 steps, final train loss _[pending]_ |

Serving: `bootstrap.sh` starts vLLM with `--enable-lora --lora-modules
triage-lora=adapters/triage-lora --max-lora-rank 16`; the adapter is addressable as
`model="triage-lora"` alongside the base model.

---

## 7. Results

### 7.1 Isolated extraction — `data/eval/extraction_golden.jsonl`, 112 rows

| metric | base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| row exact-match | 0.563 | 0.607 | _[pending]_ |
| key precision | 0.768 | 0.782 | _[pending]_ |
| key recall | 0.790 | 0.833 | _[pending]_ |
| key F1 | 0.779 | 0.807 | _[pending]_ |
| value accuracy | 0.844 | 0.826 | _[pending]_ |
| keys missed (FN) | 29 | 23 | _[pending]_ |
| keys hallucinated (FP) | 33 | 32 | _[pending]_ |
| rows with a hallucinated slot | 27 | 24 | _[pending]_ |

v1: recovers missed slots (recall +4.3 pts, exact-match +4.5 pts) at a small value cost
(−1.8 pts; a new "50 → age" bystander error). Hallucination count barely moved.
v2 goal: value accuracy ≥ 0.844 while holding key-F1.

### 7.2 Isolated distress — derived from the 59 scenarios, 6 flags

| metric | base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| flag exact-match | 0.915 | 0.915 | _[pending]_ |
| micro precision | 0.636 | 0.571 | _[pending]_ |
| micro recall | 0.875 | 1.000 | _[pending]_ |
| micro F1 | 0.737 | 0.727 | _[pending]_ |
| scenarios missing a needed flag | 1 | 0 | _[pending]_ |
| scenarios with a spurious flag | 4 | 5 | _[pending]_ |

Per-flag F1, base → v1: `severe_difficulty_breathing` 1.00→1.00,
`confused_or_hard_to_awaken` 1.00→1.00, `shock_signs` 1.00→0.67 (one new spurious fire),
`visible_facial_diaphoresis` 1.00→1.00,
`triager_assessment_life_threatening` 0.00→0.29 (base missed the needed one *and* fired
4 spurious; v1 catches the needed one, fires 5 spurious),
`triager_assessment_very_sick_weak` 1.00→1.00.

**Caveat on the "spurious" flags.** The derived gold only credits the one flag each
scenario was written to test. The extra `life_threatening` firings are on
`ems: severe dyspnea`, `ems: shock signs`, `ems: syncope` — presentations that *are*
life-threatening and route to `CALL_EMS_911` regardless. So the measured precision drop
overstates the real over-triage risk; a stricter reading is "v1 raised recall to 1.0 and
added no disposition-changing false positive."

### 7.3 End-to-end (secondary) — `data/eval/scenarios.jsonl`, 59 scenarios

| metric | baseline | LoRA v1 | LoRA v2 |
|---|---|---|---|
| triage accuracy | 0.915 | 0.915 | _[pending]_ |
| workflow accuracy | 0.915 | 0.915 | _[pending]_ |
| under-triage rate | 0.034 | 0.051 | _[pending]_ |
| over-triage rate | 0.051 | 0.034 | _[pending]_ |
| red-flag recall (911) | 0.944 | 0.889 | _[pending]_ |
| red-flag recall (top-2 tier) | 0.970 | 0.939 | _[pending]_ |
| call total (s), p50 | 3.38 | 2.31 | _[pending]_ |

v1 is **flat on accuracy** and the mix shifts: over-triage down, under-triage and
red-flag recall down. The isolated evals show the model *did* improve — the end-to-end
number is gated by §7.4.

### 7.4 The pipeline defect that gates the end-to-end result

**`classify_distress` runs on every turn**, and its schema requires `life_threatening` /
`very_sick_or_weak` as booleans — so every turn it emits one. In `agent.py`:

```python
merged.update(distress.data or {})   # {"triager_assessment_life_threatening": False} on a calm turn
turn = self.fsm.ingest(merged)       # ingest keeps any non-None value  →  slot = False
```

A patient who leads with panic ("this is an emergency") and then answers follow-ups
calmly produces `life_threatening=True` on turn 1 and `False` on turn 3, and the later
value **overwrites** the earlier one. By the time the FSM finalizes, the flag is gone →
the scenario falls through to `HOME_CARE`.

**Evidence** (`code/probe_distress.py --scenarios`): on the real turn-1 utterance for
`ems: triager judges life-threatening`, base returns *all-false* but LoRA v1 correctly
returns `life_threatening` — yet the scenario still ends `HOME_CARE`. The flag is raised
and then cleared.

**Attempted fix, reverted.** Making the six observed flags monotonic across turns
(never `True → False`) sent baseline **over-triage from 0.051 to 0.339** — revealing
that the overwrite was silently walking back ~15 classifier false positives per run. The
distress classifier's *precision* is not high enough for stickiness to be safe; the
overwrite is load-bearing. Reverted in `00ff639`; the correct fix (call the classifier
on the chief-complaint turn only, or gate stickiness on classifier confidence) is left
as future work.

This is why two of the five failures (`ems/educ: triager judges …`) cannot be fixed by
any adapter: the schema forces a boolean every turn, and the pipeline treats the last
one as authoritative.

---

## 8. Incidental bugs fixed

| bug | effect | fix |
|---|---|---|
| `finetune_lora.py` / `bootstrap.sh` defaulted `--model` to `Qwen/Qwen2.5-7B-Instruct` while serving `Llama-3.1-8B-Instruct` | a run without `--model` trains an adapter vLLM can't load against the served base | default → `meta-llama/Llama-3.1-8B-Instruct` in all three places (`d2aa28d`) |
| `--max-seq-len` default 1024; `EXTRACT_SYS` alone is ~1,640 tokens | every extraction training row's gold label was truncated off the end of the sequence | default → 2048; added a token-length histogram + truncation warning after dataset build (`d2aa28d`) |
| `device_map="auto"` in `from_pretrained` | when a vLLM server still held the GPU, accelerate spilled layers to CPU and bitsandbytes 4-bit aborted with an opaque error | pin to `{"": 0}` — fits or raises a clean CUDA OOM (`b6f83fc`) |

---

## 9. Limitations & next steps

1. **Fix the distress-merge defect** (§7.4) — the single highest-value change; likely
   recovers 2–3 failures with no fine-tuning and lifts red-flag recall to 1.0.
2. **The eval is thin** — 59 scenarios, 112 golden rows, and a *deterministic templated*
   simulated patient (`patient_sim.py`). A LoRA that improves templated phrasings may be
   partly fitting the template. A larger, non-templated eval (and a held-out split)
   should precede any further fine-tuning.
3. **Distress-classifier precision** — the real weakness. Worth a dedicated
   precision-focused pass (more hard negatives) once (1) is done and stickiness is safe.
4. **Consider a stronger base model** — for a 91.5% system, a larger or hosted model may
   beat fine-tuning an 8B, latency budget permitting.

---

## 10. Reproduction

```bash
# --- pod setup ---
export HF_TOKEN=hf_...
bash bootstrap.sh                                  # venv + pinned deps + serve base model

# --- baseline (isolated + end-to-end) ---
python code/eval_extraction.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/extract_base.json
python code/eval_distress.py   --model meta-llama/Llama-3.1-8B-Instruct --out outputs/distress_base.json
python code/eval_harness.py --out outputs/eval_baseline_all.json --label baseline

# --- train ---
pkill -9 -f "vllm serve"; sleep 3                  # training needs the GPU to itself
python code/generate_sft_set.py                    # -> data/train/sft_v1.jsonl (290 rows)
python code/finetune_lora.py --epochs 5            # -> adapters/triage-lora/

# --- serve adapter + re-eval ---
SKIP_INSTALL=1 LORA_DIR=adapters/triage-lora bash bootstrap.sh
python code/eval_extraction.py --model triage-lora --out outputs/extract_lora.json
python code/eval_extraction.py --compare outputs/extract_base.json outputs/extract_lora.json
python code/eval_distress.py   --model triage-lora --out outputs/distress_lora.json
python code/eval_distress.py   --compare outputs/distress_base.json outputs/distress_lora.json
TRIAGE_MODEL=triage-lora python code/eval_harness.py --out outputs/eval_lora.json --label triage-lora
```

---

## 11. File inventory (this branch)

| file | purpose |
|---|---|
| `code/generate_sft_set.py` | deterministic SFT builder |
| `data/train/sft_v1.jsonl` | 290-row training set |
| `data/train/README.md` | the three datasets + run steps |
| `code/eval_extraction.py` | isolated extractor eval (per-slot P/R/F1, hallucinations) |
| `code/eval_distress.py` | isolated distress-classifier eval (per-flag P/R/F1) |
| `code/probe_distress.py` | base-vs-adapter distress diff on hard phrases / real scenarios |
| `tests/test_sft_set.py` | training-file guard tests |
| `code/finetune_lora.py` | QLoRA trainer (model default, max-seq-len, device_map fixes) |
| `bootstrap.sh` | pod setup + vLLM serve (model default fix) |
| `outputs/eval_baseline_all.json` | baseline end-to-end |
| `outputs/extract_*.json`, `outputs/distress_*.json` | isolated eval results |
