# tests/audit/

Protocol-conformance artifacts — **not** pytest. These check `code/rules.yaml`
against the source guideline PDF, separately from the unit tests in `tests/`.

| file | purpose |
|---|---|
| `protocol_index.yaml` | the source guideline's disposition tiers and rules, transcribed by hand from the PDF |
| `golden_cases.yaml` | worked triage cases with the disposition the PDF prescribes |
| `check_rules.py` | 7 checks of `rules.yaml` vs `protocol_index.yaml`: coverage (no missing rules), orphans (no invented rules), tier match, care-advice code lists, disposition ordering, atom/operator structure, and reachability (no state yields "no disposition") |
| `run_golden.py` | runs `golden_cases.yaml` through the decision engine and reports any mismatch vs the PDF-prescribed disposition |

Run: `python tests/audit/check_rules.py` and `python tests/audit/run_golden.py`.
The pytest suite (`pytest -q`) covers engine/state-machine/guardrail behaviour;
this directory covers "does the encoded table faithfully represent the
guideline".
