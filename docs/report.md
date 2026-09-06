# Chest-Pain Triage Chat Agent — Project Report

**Status:** fine-tuning results final. CPU-latency row and demo video pending
(marked `_[pending]_`).

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
| Baseline triage accuracy (59 scenarios) | **0.915**  (95% CI [0.83, 0.98]) |
| Baseline red-flag recall (gold = 911) | 0.944  (95% CI [0.81, 1.00]) |
| Baseline call latency (GPU) | TTFT p50 0.32 s · total p50 3.38 s |
| LoRA vs baseline — every metric (isolated + end-to-end) | difference CI includes 0 |

**The fine-tune produced no statistically detectable change** on any metric — slot
extraction, distress classification, or end-to-end disposition — at this eval size.
Not an improvement, not a regression. This is the *predicted* outcome: the
deterministic engine owns the safety-critical disposition, and the LLM only extracts
facts and phrases questions — tasks the base 8 B already does near-zero-shot. The
adapter is **not shipped** (it adds serving cost and version management for no measured
benefit).

The higher-value lever found during this work is a **pipeline defect** in how the
distress flags merge across turns (§6.4), not fine-tuning.

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
may well suffice — quantified as future work (§8).

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
| `data/train/sft_v1.jsonl` | 290 | one LLM call (`extract_slots` *or* `classify_distress`) + gold JSON, in chat form | **SFT/LoRA training** (the reported v1 adapter used an earlier 268-row version — §6.1) |
| `data/eval/extraction_val.jsonl` / `_test.jsonl` | 25 / 87 | the golden set split by `tag` for future model selection (`code/split_val.py`) | validation / held-out test |

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
memorisation. The generator was revised twice while iterating (v1 → v2 → v3, §6.1); the
file currently in the repo is the v3 content (290 rows: 219 extract / 71 distress). The
reported v1 adapter used the earlier 268-row version at commit `d2aa28d`. See
`data/train/README.md` for the composition breakdown.

### 4.3 Train / validation / evaluation separation

- **Training:** `data/train/sft_v1.jsonl` — synthetic, generated by a process
  independent of the eval sets (no shared rows).
- **Evaluation (held-out):** the 59 scenarios and 112 golden rows, reported in §5–§6.
- **Validation:** `code/split_val.py` produces `data/eval/extraction_val.jsonl` (25 rows
  / 20 tags) and `extraction_test.jsonl` (87 / 77), split by `tag` so no template
  straddles. This arrived after the v1 comparison, which therefore used the full 112-row
  golden set — acceptable here because v1 shows no measurable effect on either split,
  but the val set is the right instrument for any future selection. Hyperparameters
  followed established QLoRA ranges (r ∈ {8,16,32}, α = 2r, lr ∈ {1e-4, 2e-4}) rather
  than a sweep.

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
| wall-clock / scenario (s) — GPU | 52.0 | 52.6 | 77.6 |

(Wall-clock per scenario includes ~7 sequential turns × 2 LLM calls each, run
8-way concurrent across scenarios.)

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
| **v1** (reported) | SFT set @ `d2aa28d`, 268 rows, 3 epochs, final train loss 0.662 |
| v2 (rolled back) | true-only distress output format + 5 epochs → regressed the distress classifier (isolated micro-F1 ≈ 0.29, optional-flag recall → 0) and over-fit the extractor (guard-slot key-F1 1.0 → 0.67). Format + epoch changed together, so cause not isolated; reverted in the dataset generator. |
| v3 (not measured) | reverted the distress format, kept the extraction rows, dropped to 3 epochs. Could not be measured cleanly — the serving environment had degraded to where the *base* model itself scored 0.57 key-F1 on a previously-0.78 eval (see §8). |

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

All differences below are **paired percentile bootstrap** (B = 5000) over the shared
rows (`code/eval_ci.py`). "Beyond noise?" = does the 95% CI on the difference exclude
zero. Both models ran on a verified-healthy server (checked via the hallucinated-key
count in the saved rows: 33 FP for base, 32 for LoRA — the healthy signature; a degraded
run shows ~175, see §8).

**Isolated slot extraction — 112 golden rows**

| metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|
| row exact-match | 0.562 | 0.607 | +0.045 | [−0.036, +0.125] | **no** |
| key precision | 0.768 | 0.782 | +0.015 | [−0.056, +0.090] | **no** |
| key recall | 0.790 | 0.833 | +0.043 | [−0.022, +0.115] | **no** |
| key F1 | 0.779 | 0.807 | +0.028 | [−0.030, +0.098] | **no** |
| value accuracy | 0.844 | 0.826 | −0.018 | [−0.080, +0.044] | **no** |
| hallucination row-rate | 0.241 | 0.214 | −0.027 | [−0.098, +0.045] | **no** |

**Isolated distress classification — 59 utterances, 6 flags**

| metric | base | LoRA v1 | Δ | 95% CI on Δ | beyond noise? |
|---|---|---|---|---|---|
| micro precision | 0.636 | 0.571 | −0.065 | [−0.208, +0.069] | **no** |
| micro recall | 0.875 | 1.000 | +0.125 | [0.000, +0.417] | **no** |
| micro F1 | 0.737 | 0.727 | −0.010 | [−0.156, +0.196] | **no** |
| flag exact-match | 0.915 | 0.915 | 0.000 | [−0.051, +0.051] | **no** |

**End-to-end (secondary) — 59 scenarios**

| metric | baseline | LoRA v1 | note |
|---|---|---|---|
| triage accuracy | 0.915 | 0.915 | within baseline CI [0.83, 0.98] |
| workflow accuracy | 0.915 | 0.915 | " |
| red-flag recall (911) | 0.944 | 0.889 | within baseline CI [0.81, 1.00] |
| under-triage rate | 0.034 | 0.051 | within baseline CI [0.00, 0.09] |
| over-triage rate | 0.051 | 0.034 | within baseline CI [0.00, 0.12] |
| call total (s), p50 | 3.38 | 2.31 | — |

**Reading.** Every metric — extraction, distress, and end-to-end — has a 95% CI on the
base-vs-LoRA difference that **includes zero**. Point estimates lean slightly positive
on extraction *recall* (a few more slots recovered) and slightly negative on extraction
*precision* and distress precision, but none survives the CI at n = 59 / 112. **There is
no measurable effect.**

The distress-recall point estimate (0.875 → 1.000) is the most suggestive single number
— the LoRA misses no needed flag in this set — but its CI runs [0.000, +0.417], five of
the six flags have support = 1, and §6.4 shows even a correctly-raised flag does not
reach a disposition. It is not a reliable signal.

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
stickiness on classifier confidence) is future work (§8).

This is why two of the five failures cannot be fixed by any adapter — the schema forces
a boolean every turn and the pipeline treats the last one as authoritative.

### 6.5 Ship / no-ship

Decision rule (pre-registered): ship the adapter only if it beats baseline on the
metrics that matter — extraction validity and red-flag behaviour — by **more than the
confidence interval**.

**Verdict: do not ship.** No metric's difference-CI excludes zero, so the criterion is
not met. Shipping a no-effect adapter adds a second serving artifact, a `--max-lora-rank`
constraint, and version management, for no measured benefit. "The fine-tune was not
justified by the eval" is a legitimate — and here, predicted — outcome for a
rules-engine architecture where the model only extracts facts and phrases questions.

What *would* change the picture: a larger, non-templated eval (real de-identified
transcripts) where the extraction-recall lean could become significant, and fixing §6.4
so a recovered distress flag can actually reach a disposition. Both are future work (§8).

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
2. **Validation split arrived late** (§4.3) — `data/eval/extraction_val.jsonl` (25 rows
   / 20 tags) and `extraction_test.jsonl` (87 / 77), split by `tag` via
   `code/split_val.py`, exist now for future selection. The v1-vs-baseline comparison in
   §6.3 predates it and uses the full 112-row golden set; since v1 shows no measurable
   effect either way, this does not change the conclusion.
3. **Thin eval** — 59 scenarios, 112 golden rows, deterministic templated patient
   (`patient_sim.py`). Several distress flags have support = 1. CIs are wide (triage
   accuracy ±~8 pts). A larger, non-templated eval — ideally de-identified real
   transcripts with clinician-adjudicated labels — is the honest next step, and the
   setting in which the extraction-recall lean might become significant.
4. **Unconstrained decoding** — guided/schema-enforced decoding was removed from
   `llm_client._chat` (commit `b4916df`) to resolve request timeouts; all results here
   used free-form generation + regex JSON extraction. Fair for base-vs-adapter (both
   identical), but schema conformance is not enforced at serving time. Re-enable once a
   backend handles the distress schema without stalling.
5. **Recurring serving instability** — observed three times on this vLLM 0.6.3 +
   single-GPU setup. (a) During baseline collection, individual model calls on an idle
   server ranged ~3 s → 300+ s with no error; a clean vLLM restart cleared it. (b) After
   the v2/v3 training cycles the *base* model's isolated extraction key-F1 fell from 0.78
   to 0.57 with a 5× jump in hallucinated keys, on an identical eval, and did **not**
   recover on a soft restart — the pod could not be hard-restarted, so v3 was not
   measurable. Not root-caused; plausibly GPU-memory or KV/prefix-cache state that
   survives `pkill`. Mitigations: hard-restart the instance between train/serve cycles;
   discard any run whose `call_total_seconds` p90 is anomalous or whose *base* canary
   score has drifted, and re-collect on a fresh server. The reported baseline and v1
   numbers were collected before (b) and verified against the healthy-server signature
   (§6.3).
6. **Fine-tune experimental hygiene** — v1 is the reported result. v2 changed the
   distress output format *and* the epoch count together, so its regression cannot be
   attributed to one variable; v3 (which isolated the epoch change) could not be
   measured (see 5). No held-out capability probe was run for catastrophic forgetting,
   and no LR/rank sweep — hyperparameters followed established QLoRA ranges.
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
