# outputs/iterations/

Result JSONs from a fine-tuning iteration (**v2**) that was measured, regressed,
and rolled back — kept as evidence for the claims in the report PDF §6.

The **reported** result is one level up: `../eval_lora.json`,
`../extract_lora.json`, `../distress_lora.json` (LoRA v1).

| file | run | outcome |
|---|---|---|
| `extract_lora_v2.json` · `distress_lora_v2.json` · `eval_lora_v2.json` | v2 — true-only distress output format + 5 epochs | distress isolated micro-F1 ≈ 0.29; guard-slot key-F1 1.0 → 0.67. Reverted in the dataset generator. |
