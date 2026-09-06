# Chest-Pain Triage Chat Agent

A multi-turn chat agent that triages adult, non-traumatic chest pain to one of the
Schmitt-Thompson *"Chest Pain – After Hours Telehealth Triage Guidelines | Adult |
2026"* dispositions.

**Hybrid design:** a deterministic rules engine makes every triage decision; a local
`meta-llama/Llama-3.1-8B-Instruct` model is used only to extract structured facts from
free text, classify how distressed the patient sounds, and phrase the questions. **It
never chooses a disposition and never diagnoses** — the final instruction is templated
protocol text.

**Full write-up:** [`docs/Chest-Pain-Triage-Agent-report.pdf`](docs/Chest-Pain-Triage-Agent-report.pdf)
— pipeline, model choice, evaluation, fine-tuning (baseline vs LoRA), safety, limitations.

## Install & run

```
pip install -r requirements.txt && pytest -q
```

One command: installs the core dependencies and runs the 205-test suite. No GPU or
network required — the rules engine, state machine, guardrails, and eval harness all
run offline against a stubbed LLM.

The **live LLM agent** needs an OpenAI-compatible server (a GPU). On a CUDA box:

```
export HF_TOKEN=hf_...        # Llama-3.1-8B-Instruct is gated
bash bootstrap.sh            # venv + full training/serving deps + vLLM on :8000
python code/app.py           # Gradio chat UI   (headless alternative: python code/chat_cli.py)
```

`requirements.txt` covers the offline core, tests, and UI. The training/serving stack
(torch, transformers, peft, trl, vllm, …) is installed by `bootstrap.sh`;
`requirements.lock` is the exact frozen environment that produced the reported numbers.

## Layout

| path | contents |
|---|---|
| `code/` | pipeline (`rules.yaml`, `decision_engine.py`, `state_machine.py`, `slots.py`, `llm_client.py`, `guardrails.py`, `agent.py`), chat UI (`app.py`, `chat_cli.py`), eval harness, SFT generator, QLoRA trainer, isolated-eval + bootstrap-CI tools |
| `data/eval/` | 59 end-to-end scenarios · 112 extraction golden rows · template-split val/test |
| `data/train/` | 290-row synthetic SFT set + generation notes |
| `outputs/` | `accuracy_report.md` + baseline & LoRA eval JSONs (`iterations/` = rolled-back attempts) |
| `docs/` | the report PDF (+ longer-form markdown backing it) |
| `tests/` | `pytest -q` → 205 passing; `tests/audit/` = protocol-vs-PDF conformance checks |

Evaluation and fine-tune commands are in the report PDF §5–§6.
