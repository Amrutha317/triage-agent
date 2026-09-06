# Chest-Pain Triage Chat Agent

Multi-turn chat triage agent for **adult, non-traumatic chest pain**, driven
by the Schmitt-Thompson *"Chest Pain – After Hours Telehealth Triage
Guidelines | Adult | 2026"* protocol.

Hybrid design: a **deterministic rules engine + state machine own every
triage decision**; a local open-source LLM (≤ 20B) only (a) extracts
structured slots from free-text patient replies and (b) phrases the agent's
questions and final instructions. **The LLM never chooses a disposition and
never diagnoses.**

## Status

| Component | State |
|---|---|
| `code/rules.yaml` + `code/decision_engine.py` + `code/slots.py` — protocol decision table, 3-valued evaluator, slot schema | ✅ |
| `code/state_machine.py` — next-question selection, red-flag short-circuit, fallback | ✅ |
| `code/llm_client.py` — slot extractor + distress classifier + question NLG | ✅ |
| `code/guardrails.py` — output post-filter (no diagnosis, no unbacked reassurance) | ✅ |
| `code/agent.py` + `code/app.py` — conversation wiring + Gradio chat UI | ✅ |
| `code/eval_harness.py` + `data/eval/` — triage/workflow accuracy, TTFT, latency | ✅ |
| `data/train/` + `code/generate_sft_set.py` + `code/finetune_lora.py` — SFT set + QLoRA | ✅ |
| baseline-vs-LoRA report + CIs (`code/eval_extraction.py`, `eval_distress.py`, `eval_ci.py`) | ✅ |
| `tests/` — `pytest -q` → **205 passing** | ✅ |
| CPU-latency row · demo video | ⏳ |

## Quick start

```
pip install -r requirements.txt      # deterministic core + eval client + UI
pytest -q                            # 205 tests, no GPU / network needed
```

Everything above the LLM runs offline. To exercise the full LLM pipeline you need a
vLLM server (a GPU pod — see below).

## Running the full pipeline (GPU pod)

```
export HF_TOKEN=hf_...                                  # Llama-3.1-8B is gated
bash bootstrap.sh                                       # .venv + pinned deps + vLLM on :8000
python code/app.py                                      # Gradio chat UI
python code/eval_harness.py --out outputs/eval_baseline_all.json --label baseline
```

Fine-tune + compare (full sequence in `docs/report.md` §9):

```
python code/generate_sft_set.py                         # -> data/train/sft_v1.jsonl
python code/finetune_lora.py                            # -> adapters/triage-lora/  (gitignored)
LORA_DIR=adapters/triage-lora bash bootstrap.sh
python code/eval_extraction.py --model triage-lora --out outputs/extract_lora.json
python code/eval_ci.py outputs/extract_base.json outputs/extract_lora.json
```

The trained adapter is **not committed** (large binary; and the eval found no
measurable benefit — `docs/finetuning-report.md`). To regenerate the exact reported
v1 adapter: `git checkout d2aa28d -- code/generate_sft_set.py` then the two commands
above.

## Model selection

| Role | Model | Why |
|---|---|---|
| **Primary** | `meta-llama/Llama-3.1-8B-Instruct` | Strong instruction-following + constrained-JSON extraction at 8B; open weights; ≤ 20B; fits one 24 GB GPU for both vLLM serving and QLoRA. |
| Low-latency comparison | `microsoft/Phi-3.5-mini-instruct` (3.8B) | TTFT / cost floor. _Comparison run not completed._ |
| Alt. extractor | `Qwen2.5-7B-Instruct` | Very reliable JSON-schema adherence if Llama extraction is noisy. |

GPU work runs on a RunPod instance (`bootstrap.sh` provisions the venv, pins the full
dependency lattice, and starts vLLM as an OpenAI-compatible server). A CPU-latency
comparison row is outstanding.

## Reports (the deliverable document)

| file | contents |
|---|---|
| `docs/report.md` | pipeline architecture, model selection, datasets, baseline accuracy + latency, fine-tuning (baseline vs LoRA, with confidence intervals), safety-rule enforcement, limitations |
| `outputs/accuracy_report.md` | the numbers + verdict, condensed |
| `docs/finetuning-report.md` | fine-tuning deep dive: dataset design, QLoRA config, the two failed iterations, the distress-merge pipeline defect, serving-instability notes |

**Headline result:** baseline triage accuracy 0.915 (95% CI [0.83, 0.98]); every
base-vs-LoRA difference — extraction, distress, end-to-end — has a difference-CI that
includes zero, i.e. **no statistically detectable effect**, the predicted outcome for
an architecture where a deterministic engine owns the disposition.

## Documented assumptions

The protocol excerpt leaves some things unspecified or tense-ambiguous.
Each is resolved conservatively (route **up**, never down) and pinned by a
test.

- **A1 — "chest pain lasts > 5 min" tense.** The four prolonged-pain EMS-911
  rules additionally require `chest_pain_present_now == true`. Without this,
  every lower tier that begins *"occurred in the past 3 days"* / *"resolved
  more than 3 days ago"* would be unreachable. A resolved > 5-min episode
  routes to **ED/UCC** (within 72 h) or **See PCP within 24 h** (older),
  exactly as the protocol's own lower tiers describe.
- **A2 — pregnancy / L&D.** The "Go to L&D Now" branching logic is not in
  this excerpt. Pregnancy does **not** change the computed disposition in
  v1; the engine attaches `pregnant = true` to its result and the NLG layer
  adds "an ED with labor-and-delivery capability is appropriate" to any
  ED-or-higher instruction.
- **A3 — duration buckets.** `duration` is a 3-way exclusive enum
  `few_seconds | under_5_min | over_5_min`. The protocol's recurring
  *"Exception: chest pains that last only a few seconds"* is then just
  `duration == few_seconds`.
- **A4 — the two "triager judgment" rules** (*"sounds like a life-threatening
  emergency"*, *"patient sounds very sick or weak"*). Encoded at their native
  tiers with a `triager_judgment` flag. The corresponding slots are set
  **only** by a standalone LLM distress classifier (never by slot
  extraction), and every firing is logged for the accuracy report's
  known-limitations section.
- **A5 — cardiac risk-factor count.** `cardiac_risk_factors` counts only
  diabetes, hypertension, high cholesterol, obesity (BMI ≥ 30), smoking,
  peripheral vascular disease, and strong family history. Age is scored
  separately by the `age > 44` / `age > 30` rules; known CAD (prior MI,
  stent, CABG, diagnosed angina, takes nitroglycerin) is captured by
  `history_of_heart_disease`.
- **Exception clauses are optimistic.** A protocol "Exception:" blocks its
  rule only when the exception is *definitely* true. An unknown exception is
  treated as not applying, so a missing answer never downgrades urgency.
  `state_machine.py` still asks the exception question when time allows.
- **Shadowed rule.** `pcp3_stable_angina_nitro_resolved` (See PCP 3 days) is
  reachable only while a cardiac symptom still persists; when the pain is
  fully resolved the higher `pcp24_brief_pain_resolved` (24 h) rule matches
  first. This is protocol-faithful (24 h is listed above 3 days) and both are
  conservative PCP referrals.

## Safety properties enforced by the core

- No path concludes a diagnosis. `rule_out` lists are differential
  considerations for the receiving clinician, never shown to the patient as
  conclusions.
- **GERD is concluded by exactly one rule** (`cpcp24_gerd`) and only when all
  three of its conditions — matches previously diagnosed heartburn **and**
  burning in chest **and** sour taste in mouth — are explicitly true. It is
  never a default or a fallback.
- Red-flag rules sit at the top of the ordered table, so the state machine
  can stop questioning and escalate the moment one is definitively true.
