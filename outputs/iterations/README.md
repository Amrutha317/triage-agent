# outputs/iterations/

Result JSONs from fine-tuning iterations that were **not** adopted. Kept as
evidence for the claims in `docs/report.md` §6.1; not part of the reported
result (that is `../eval_lora.json`, `../extract_lora.json`,
`../distress_lora.json` = LoRA v1).

| file | run | outcome |
|---|---|---|
| `extract_lora_v2.json` / `distress_lora_v2.json` / `eval_lora_v2.json` | **v2** — true-only distress output format + 5 epochs | regressed: distress isolated micro-F1 ≈ 0.29, guard-slot key-F1 1.0 → 0.67. Rolled back in the dataset generator. |

A later **v3** (distress format reverted, 3 epochs) could not be measured — the
serving environment degraded past a usable base-model canary; its numbers are
not reported and not kept.
