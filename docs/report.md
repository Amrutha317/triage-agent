# Chest-Pain Triage Chat Agent — Project Report

**Status:** draft — baseline + LoRA v1 results complete; LoRA v2 (5-epoch retrain),
CPU-latency row, and the demo video are pending and marked `_[pending]_` below.

**Assignment:** *Triage Chat Agent Development Using Open-Source LLM* (5-day take-home).
**Repo branch:** `sft-dataset-and-model-fix` · **PR:** #1
**Protocol:** *Chest Pain – After Hours Telehealth Triage Guidelines | Adult | 2026*
(Schmitt-Thompson).

---

## 1. Summary

A multi-turn chat agent that triages adult, non-traumatic chest pain to one of the
protocol's seven dispositions. It is a **hybrid**: a deterministic rules engine owns
every triage decision; a local ≤ 20 B open-source LLM only (a) extracts structured facts
from free-text replies, (b) classifies how distressed the patient sounds, and (c)
phrases questions. **The LLM never chooses a disposition and never diagnoses.**

| | result |
|---|---|
| Baseline triage accuracy (59 scenarios) | **0.915** |
| Baseline red-flag recall (gold = 911) | 0.944 |
| Baseline call latency (GPU) | TTFT p50 0.32 s · total p50 3.38 s |
| LoRA v1 vs baseline — disposition | flat (within noise) |
| LoRA v1 vs baseline — slot extraction key-F1 | 0.779 → **0.807** |
| LoRA v1 vs baseline — distress recall | 0.875 → **1.000** (no missed emergency) |
| LoRA v2 | _[pending]_ |

The fine-tune helps the layer it targets (extraction, distress recall) and is neutral
on end-to-end disposition — expected for a rules-engine architecture, and consistent
with the pre-registered position that "LoRA mainly improves extraction/format, not the
disposition the engine owns." One end-to-end pipeline defect that blocks the distress
gain from reaching a disposition is identified and analysed in §6.4.

---

## 2. Pipeline architecture

```
patient text ──┬─► llm_client.classify_distress()  ─► 6 observed red-flag / triager slots
               └─► llm_client.extract_slots()       ─► {slot: value}  (schema from slots.py)
                        │  merge into conversation state
                        ▼
               state_machine.ingest(state)
                        │  runs decision_engine on everything known so far
                        ├─ a rule fires ─► escalate now (after ≤1 exception-confirm
                        │                   question; pregnancy question for ED tiers;
                        │                   EMS-911 stays short)
                        └─ no rule fires ─► ask the question-group the most-severe
                                            still-reachable rule needs
                        ▼
               llm_client.render(turn)          ─► question phrased by the LLM;
                                                   disposition sentence is TEMPLATED
                        ▼
               guardrails.guard_question / guard_final  ─► text shown to patient
```

| module | role | LLM? |
|---|---|---|
| `code/rules.yaml` | protocol as an ordered decision table (8 tiers, ~30 rules, metadata per rule) | no |
| `code/decision_engine.py` | 3-valued-logic evaluator; first definitively-true rule wins | no |
| `code/state_machine.py` | next-question selection, red-flag short-circuit, fallback | no |
| `code/slots.py` | slot schema, ask order, extractor JSON schema | no |
| `code/llm_client.py` | `extract_slots`, `classify_distress`, question NLG | **yes** |
| `code/guardrails.py` | output post-filter (no diagnosis, no unbacked reassurance) | no |
| `code/agent.py` | per-conversation wiring of the above | no |
| `code/app.py` | Gradio chat UI + per-turn latency logging | — |
| `code/patient_sim.py` | deterministic simulated patient for eval | no |
| `code/eval_harness.py` | scenario runner + metrics | no |

**Why the LLM does not choose the disposition.** The disposition is the safety-critical
output. An LLM is non-deterministic, non-auditable, and movable by paraphrase or
injection; phrasing like "probably reflux" can pull it toward under-triage. With a rules
engine, the disposition is a pure function of the protocol table and the extracted
slots, unit-tested rule-by-rule (`tests/` — 200 passing), and every final decision
records the exact rule that fired for post-incident replay.

**Three-valued logic.** Every rule atom is TRUE / FALSE / UNKNOWN; a rule fires only on
a definite TRUE. A missing slot makes dependent rules UNKNOWN, which (a) lets the state
machine short-circuit the instant a red flag is definitely true and (b) tells it which
slot to ask next. "Missing = false" would silently rule out red flags — the wrong
direction.

---

## 3. Model selection

| role | model | reasoning |
|---|---|---|
| **Primary** | `meta-llama/Llama-3.1-8B-Instruct` | Strong instruction-following and constrained-JSON extraction at 8 B; fits one 24 GB GPU for both vLLM serving and QLoRA training; open weights; ≤ 20 B per the constraint. |
| Low-latency comparison | `microsoft/Phi-3.5-mini-instruct` (3.8 B) | TTFT / cost floor; quantifies the accuracy trade-off. _[comparison run pending]_ |
| Alternative extractor | `Qwen2.5-7B-Instruct` | Very reliable JSON-schema adherence if Llama extraction is noisy. |

All candidates are open-weight and ≤ 20 B. Serving is vLLM 0.6.3 (OpenAI-compatible),
which also serves the LoRA adapter without merging. The 8 B choice is deliberately
conservative: because extraction is schema-shaped and NLG is templated, a smaller model
may well suffice — quantified as future work (§7).

> **Note.** `finetune_lora.py` and `bootstrap.sh` originally defaulted to
> `Qwen/Qwen2.5-7B-Instruct` while serving Llama; a run without an explicit `--model`
> would have trained an adapter the served base could not load. Fixed — all three
> places now default to Llama-3.1-8B-Instruct.

---

## 4. Synthetic datasets

Three artifacts, three roles:

| file | rows | one row is | role |
|---|---|---|---|
| `data/eval/scenarios.jsonl` | 59 | full patient case (`facts`) + gold disposition + gold rule_id | **end-to-end evaluation** |
| `data/eval/extraction_golden.jsonl` | 112 | one patient utterance + the exact `{slot: value}` a perfect extractor returns | **isolated extraction evaluation** |
| `data/train/sft_v1.jsonl` | 290 | one LLM call (`extract_slots` *or* `classify_distress`) + gold JSON, in chat form | **SFT/LoRA training** |

### 4.1 How the eval data is labelled

Each scenario's gold disposition is derived **deterministically** — its ground-truth
`facts` run through the same `rules.yaml`, spot-checked by hand against the protocol PDF.
The simulated patient (`patient_sim.py`) is deterministic and templated: it converts
`facts` into short patient-style sentences, volunteering red flags first (as a
distressed caller would), then answering exactly what the agent asks. This makes
per-turn latency measurable with no simulated-patient latency to subtract.

### 4.2 How the training data is built (`code/generate_sft_set.py`)

Deterministic, no network, no LLM. Two prompt families, each **byte-identical to
serving**: the system prompt is imported verbatim from `llm_client.py`
(`EXTRACT_SYS` / `DISTRESS_SYS`) and the user message mirrors the exact string
`LLMClient.extract_slots` / `classify_distress` builds. The canonical schema is
`slots.py`; every extraction label is passed through `llm_client.validate_slots()` at
build time and the build aborts on any invalid label. `tests/test_sft_set.py` re-checks
shape, verbatim prompts, strict-JSON responses, and schema-valid labels.

The set is **generated by a different process than the eval data** and shares no rows
with `extraction_golden.jsonl`, so the isolated eval measures generalisation, not
memorisation. Composition (v2, 290 rows: 219 extract / 71 distress) is weighted toward
the five baseline failures (§5.3); see `data/train/README.md` for the full breakdown.

### 4.3 Train / validation / evaluation separation

- **Training:** `data/train/sft_v1.jsonl` — synthetic, independent process.
- **Evaluation (held-out):** the 59 scenarios and 112 golden rows, reported in §5–§6.
- **Validation:** no separate held-out validation slice was carved out. Hyperparameters
  followed established QLoRA ranges (r ∈ {8,16,32}, α = 2r, lr ∈ {1e-4, 2e-4}) rather
  than a sweep, and v1-vs-v2 was compared on the evaluation sets. This is a known
  limitation (§7): the correct approach is a **template-level** split (by the `tag`
  field, so every template lands entirely in one split) into
  `data/eval/extraction_val.jsonl` for model selection, keeping the rest as untouched
  test. _[split pending]_

---

## 5. Evaluation — baseline

`code/eval_harness.py` drives a full simulated conversation per scenario and scores the
final disposition. Model: `meta-llama/Llama-3.1-8B-Instruct`, vLLM on a single GPU,
prefix caching on, streaming on, temperature 0 for extraction/distress.

### 5.1 Accuracy (59 scenarios)

| metric | value | definition |
|---|---|---|
| **triage accuracy** | 0.915 | exact disposition == gold |
| **workflow accuracy** | 0.915 | (disposition, rule_id) == gold — "right answer, right reason" |
| under-triage rate | 0.034 | predicted disposition **less** urgent than gold |
| over-triage rate | 0.051 | predicted disposition **more** urgent than gold |
| **red-flag recall (gold = 911)** | 0.944 | of gold `CALL_EMS_911_NOW`, fraction predicted there (17/18) |
| red-flag recall (gold ∈ {911, ED-now}) | 0.970 | fraction kept in the top-2 tier (32/33) |
| did not terminate | 0 | conversations that hit the turn cap without a disposition |

Plain accuracy is **not** the headline: error cost is asymmetric (an under-triaged MI ≠
an over-triaged muscle strain) and classes are imbalanced. Red-flag recall and
under-triage rate are the metrics that matter; target red-flag recall ≥ 0.99.

### 5.2 Latency

| | mean | p50 | p90 |
|---|---|---|---|
| call TTFT (s) — GPU | 0.79 | 0.32 | 2.15 |
| call total (s) — GPU | 3.37 | 3.38 | 6.51 |
| turns / scenario | 6.4 | 7 | 10 |
| wall-clock / scenario (s) — GPU | _[from eval JSON]_ | | |

Per-turn latency is almost entirely the two LLM calls; the engine and state machine are
microseconds. Extraction is the larger call (input carries the running context); the
distress call is short.

**GPU vs CPU.** _[pending]_ — GPU numbers above are on the RunPod instance. CPU row to
be added: either a real ~10-scenario pass (Ollama / `llama.cpp` CPU) or a labelled
estimate (8 B Q4 on CPU ≈ 5–20 s/turn depending on cores) with method. The CPU path is
a cost/latency **comparison**, not the recommended deployment.

### 5.3 The five baseline failures — all LLM-layer

The rules engine is deterministic and correct on all five in offline mode
(`eval_harness.py --offline`, perfect-extraction oracle). Every failure is an
extraction or distress-classification error.

| scenario | gold → predicted | cause | call |
|---|---|---|---|
| `ems: triager judges life-threatening` | `CALL_EMS_911` → `HOME_CARE` | distress under-called `life_threatening` on the concatenated chief complaint; flag also erased downstream (§6.4) | distress |
| `educ: triager judges very sick/weak` | `GO_TO_ED_UCC` → `HOME_CARE` | flag set on turn 1, erased downstream (§6.4) | distress |
| `ed: difficulty breathing (non-severe)` | `GO_TO_ED_NOW` → `CALL_EMS_911` | classifier promoted mild "short of breath" to `severe_difficulty_breathing` | distress |
| `educ: pleuritic pain` | `GO_TO_ED_UCC` → `CALL_EMS_911` | classifier read "hurts to breathe in" as a breathing emergency | distress |
| `pcp3: nitro-resolved` | `SEE_PCP_3_DAYS` → `GO_TO_ED_NOW` | extractor hallucinated `other_symptoms=[difficulty_breathing]` from "some of those" | extract |

---

## 6. Fine-tuning — baseline vs LoRA

### 6.1 Method

| | value |
|---|---|
| technique | **QLoRA** (4-bit nf4, double quant, bf16 compute) |
| rationale | full fine-tune of 8 B won't fit training state in 24 GB; QLoRA trains comfortably and vLLM serves the adapter unmerged; quality gap vs full LoRA is negligible for a task this narrow |
| LoRA config | r = 16, α = 32, dropout 0.05, target `q_proj,k_proj,v_proj,o_proj` |
| trainable params | 13.6 M / 8.04 B (0.17 %) |
| optimizer / schedule | paged_adamw_8bit, cosine, warmup 0.03, lr 2e-4 |
| batch | 2 × grad-accum 8 (effective 16) |
| max-seq-len | **2048** (extraction rows peak ~1,800 tokens) |
| **v1** | 268 rows, 3 epochs, ~50 steps, final train loss 0.662 |
| **v2** | 290 rows, 5 epochs, final loss _[pending]_ |

> **Note.** `--max-seq-len` defaulted to 1024; `EXTRACT_SYS` alone is ~1,640 tokens, so
> every extraction row's gold label was being truncated off the end of the training
> sequence. Fixed to 2048, plus a token-length histogram + truncation warning at
> dataset-build time.

### 6.2 Evaluation approach

The disposition is owned by the deterministic engine, so LoRA is **not** expected to
move end-to-end triage accuracy much. What it should improve is what it touches: slot
extraction and distress classification. Those are measured **in isolation** —
`code/eval_extraction.py` (112 golden rows through `extract_slots()`) and
`code/eval_distress.py` (59 chief-complaint utterances through `classify_distress()`) —
so the number reflects the model, not the pipeline. End-to-end accuracy is reported as a
**secondary** number because the brief asks for it.

### 6.3 Results

**Isolated slot extraction — 112 golden rows**

| metric | base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| row exact-match | 0.563 | **0.607** | _[pending]_ |
| key precision | 0.768 | 0.782 | _[pending]_ |
| key recall | 0.790 | **0.833** | _[pending]_ |
| key F1 | 0.779 | **0.807** | _[pending]_ |
| value accuracy | 0.844 | 0.826 | _[pending]_ |
| slots missed (FN keys) | 29 | **23** | _[pending]_ |
| slots hallucinated (FP keys) | 33 | 32 | _[pending]_ |
| rows with a hallucinated slot | 27 | 24 | _[pending]_ |

**Isolated distress classification — 59 utterances, 6 flags**

| metric | base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| flag exact-match | 0.915 | 0.915 | _[pending]_ |
| micro precision | 0.636 | 0.571 | _[pending]_ |
| micro recall | 0.875 | **1.000** | _[pending]_ |
| micro F1 | 0.737 | 0.727 | _[pending]_ |
| scenarios missing a needed flag | 1 | **0** | _[pending]_ |
| scenarios with a spurious flag | 4 | 5 | _[pending]_ |

**End-to-end (secondary) — 59 scenarios**

| metric | baseline | LoRA v1 | LoRA v2 |
|---|---|---|---|
| triage accuracy | 0.915 | 0.915 | _[pending]_ |
| workflow accuracy | 0.915 | 0.915 | _[pending]_ |
| under-triage rate | 0.034 | 0.051 | _[pending]_ |
| over-triage rate | 0.051 | 0.034 | _[pending]_ |
| red-flag recall (911) | 0.944 | 0.889 | _[pending]_ |
| red-flag recall (top-2 tier) | 0.970 | 0.939 | _[pending]_ |
| call total (s), p50 | 3.38 | 2.31 | _[pending]_ |

**Reading, v1.** Extraction: a modest, real gain on getting the right *fields* (key-F1
+0.028, exact-match +0.045, 6 fewer missed slots), at a small cost on getting the right
*values* (−0.018). Distress: **recall to 1.0** — no scenario misses a needed flag,
including `ems: triager judges life-threatening` (missed at baseline) — at the cost of
precision (−0.065). The precision drop is largely a **gold-labelling artifact**: the
"spurious" `life_threatening` firings are on `ems: severe dyspnea` / `ems: shock signs`
/ `ems: syncope`, presentations that are life-threatening and route to 911 regardless.
End-to-end accuracy is flat; the composition shifts (over-triage down, red-flag recall
down) because of §6.4.

**Thin-cell caveat.** Several distress flags have support = 1 in the 59-scenario set
(`confused_or_hard_to_awaken`, `shock_signs`, `visible_facial_diaphoresis`,
`triager_assessment_life_threatening`, `triager_assessment_very_sick_weak`). Per-flag
precision/recall on a single positive is not a stable estimate; the micro-average and
the "scenarios missing a needed flag" count are the trustworthy figures. Confidence
intervals: _[to add — bootstrap over the 59 / 112 rows]_.

### 6.4 The pipeline defect that gates the end-to-end result

`classify_distress` runs on **every** turn, and its schema requires `life_threatening` /
`very_sick_or_weak` as booleans — so every turn it emits one. In `agent.py`:

```python
merged.update(distress.data or {})   # {"triager_assessment_life_threatening": False} on a calm turn
turn = self.fsm.ingest(merged)       # ingest keeps any non-None value  →  slot = False
```

A patient who leads with panic ("this is an emergency") and then answers follow-ups
calmly produces `life_threatening=True` on turn 1 and `False` on turn 3; the later value
**overwrites** the earlier one and the scenario falls through to `HOME_CARE`.

**Evidence** (`code/probe_distress.py --scenarios`): on the real turn-1 utterance for
`ems: triager judges life-threatening`, base returns *all-false* but LoRA v1 correctly
returns `life_threatening` — yet the scenario still ends `HOME_CARE`. The flag is raised
and then cleared.

**Attempted fix, reverted.** Making the six observed flags monotonic across turns
(never `True → False`) sent baseline over-triage from 0.051 to **0.339** — the overwrite
was silently walking back ~15 classifier false positives per run. The classifier's
*precision* is not high enough for stickiness to be safe as a blanket rule. Reverted;
the correct fix (call the classifier on the chief-complaint turn only, or gate
stickiness on classifier confidence) is future work (§7).

This is why two of the five failures cannot be fixed by any adapter — the schema forces
a boolean every turn and the pipeline treats the last one as authoritative.

### 6.5 Ship / no-ship

Decision rule (pre-registered): ship the adapter only if it beats baseline on the
metrics that matter — extraction validity and red-flag behaviour — by more than the
confidence interval. _[verdict after v2; on v1 alone: **do not ship** — extraction gain
is modest and end-to-end red-flag recall regressed, gated by §6.4.]_ "The fine-tune was
not decisively justified by the eval" is a legitimate outcome for a rules-engine
architecture where the model only extracts and paraphrases.

---

## 7. Safety behaviour

| safety rule (brief) | how enforced | evidence |
|---|---|---|
| **Must not diagnose** | disposition sentences are templated from the `Decision` (`DISPOSITION_SCRIPT`); `rule_out` (differential) is never surfaced to the patient; `guardrails.py` post-filters free text for diagnosis language and unbacked reassurance | `tests/test_guardrails.py` |
| **Stop questioning and escalate on a red flag** | `state_machine._next_turn` finalizes the instant a rule fires; at most one exception-confirmation question and (for ED tiers only) a pregnancy question follow; EMS-911 finalizes immediately | `tests/test_state_machine.py`, `tests/test_reachability.py` |
| **Never assume acid reflux / heartburn** | GERD is reachable through exactly one rule (`cpcp24_gerd`) requiring **all three** of previously-diagnosed heartburn match + burning + sour taste; `suspected_cause` is `decision_relevant=False` and can terminate nothing; `EXTRACT_SYS` rule 7 forbids filling the GERD slots from "it's just my reflux" | `tests/test_decision_engine.py` (asserts 2-of-3 does not conclude GERD) |

Additional: a deterministic keyword red-flag net (`keyword_redflags`) runs on every
patient message independent of the LLM classifier, union-only (can add a red flag, never
remove one), as a belt-and-suspenders for unmistakable phrases.

---

## 8. Limitations & next steps

1. **Distress-flag overwrite defect** (§6.4) — highest-value fix; expected to recover
   2–3 failures with no fine-tuning and lift red-flag recall toward 1.0.
2. **No held-out validation split** (§4.3) — hyperparameters were not swept on a
   template-level val set; v1-vs-v2 was compared on the eval sets. Add
   `data/eval/extraction_val.jsonl` split by `tag`.
3. **Thin eval** — 59 scenarios, 112 golden rows, deterministic templated patient.
   Several distress flags have support = 1. No confidence intervals yet. A larger,
   non-templated eval (ideally de-identified real transcripts with clinician labels) is
   the honest next step.
4. **Unconstrained decoding** — guided/schema-enforced decoding was removed from
   `llm_client._chat` (commit `b4916df`) to resolve request timeouts; all results here
   used free-form generation + regex JSON extraction. Fair for base-vs-adapter (both
   identical), but schema conformance is not enforced at serving time. Re-enable once a
   backend handles the distress schema without stalling.
5. **Unresolved latency degradation** — during baseline collection, individual model
   calls on an idle, correctly-configured server ranged from ~3 s to 300+ s with no
   error; a clean vLLM restart cleared it and the final baseline was collected
   immediately after. Not root-caused. A run with an anomalous `call_total_seconds` p90
   should be discarded and repeated after a restart.
6. **Catastrophic-forgetting check** — v2 uses 5 epochs (v1 used 3) on a small set; no
   generic-instruction examples were mixed in and no general-capability probe was run.
   The forgetting risk is unmeasured.
7. **Smaller model** — with extraction schema-shaped and NLG templated, Phi-3.5-mini or
   Qwen2.5-3B may match Llama-8B at lower latency; the comparison row is not yet run.
8. **CPU latency row** — see §5.2.

---

## 9. How to run

```bash
pip install -r requirements.txt
pytest -q                                    # 200 deterministic-core + pipeline tests

# --- GPU pod: serve + evaluate ---
export HF_TOKEN=hf_...
bash bootstrap.sh                             # venv + pinned deps + vLLM serving Llama-3.1-8B

python code/eval_harness.py --out outputs/eval_baseline_all.json --label baseline
python code/eval_extraction.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/extract_base.json
python code/eval_distress.py   --model meta-llama/Llama-3.1-8B-Instruct --out outputs/distress_base.json

# --- fine-tune ---
pkill -9 -f "vllm serve"; sleep 3            # training needs the GPU to itself
python code/generate_sft_set.py              # -> data/train/sft_v1.jsonl (290 rows)
python code/finetune_lora.py --epochs 5      # -> adapters/triage-lora/

# --- serve adapter + re-evaluate ---
SKIP_INSTALL=1 LORA_DIR=adapters/triage-lora bash bootstrap.sh
python code/eval_extraction.py --model triage-lora --out outputs/extract_lora.json
python code/eval_extraction.py --compare outputs/extract_base.json outputs/extract_lora.json
python code/eval_distress.py   --model triage-lora --out outputs/distress_lora.json
python code/eval_distress.py   --compare outputs/distress_base.json outputs/distress_lora.json
TRIAGE_MODEL=triage-lora python code/eval_harness.py --out outputs/eval_lora.json --label triage-lora

# --- chat UI ---
python code/app.py                           # Gradio interface
```

Demo video: _[pending — record the full chest-pain flow through `app.py`, one benign
and one red-flag case]_.

---

## 10. Appendix

### 10.1 Deliverables map (brief → repo)

| brief item | location |
|---|---|
| hybrid LLM + state-machine pipeline | `code/` (see §2) |
| chat interface | `code/app.py` |
| eval dataset + triage/workflow accuracy, TTFT, total time, GPU/CPU latency | `data/eval/`, `code/eval_harness.py`, §5 |
| SFT/LoRA dataset | `data/train/sft_v1.jsonl`, `code/generate_sft_set.py` |
| fine-tune + baseline vs LoRA results | `code/finetune_lora.py`, §6, `outputs/` |
| accuracy report | `outputs/accuracy_report.md` |
| this document | `docs/report.md` |
| fine-tuning deep dive | `docs/finetuning-report.md` |

### 10.2 Key files added for the fine-tuning task

| file | purpose |
|---|---|
| `code/generate_sft_set.py` | deterministic SFT builder (prompts imported verbatim from `llm_client.py`) |
| `code/eval_extraction.py` | isolated extractor eval — per-slot P/R/F1, hallucination rows |
| `code/eval_distress.py` | isolated distress-classifier eval — per-flag P/R/F1 |
| `code/probe_distress.py` | base-vs-adapter distress diff on hard phrases / real scenarios |
| `tests/test_sft_set.py` | training-file guard tests |
| `data/train/README.md` | the three datasets + run steps + v1→v2 rationale |
| `docs/finetuning-report.md` | full fine-tuning analysis (this document's §6, expanded) |
