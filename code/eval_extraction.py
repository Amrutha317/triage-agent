"""
eval_extraction.py -- measure the slot EXTRACTOR in isolation.

Runs data/eval/extraction_golden.jsonl (112 hand-labeled "one patient utterance
-> exact slot dict" rows) through LLMClient.extract_slots() and scores the
output. No state machine, no rules engine, no disposition -- this is the direct
measure of "did the model read the patient's words into the right fields",
which end-to-end scenario accuracy only reflects indirectly (and, right now,
through a distress-flag pipeline bug that muddies it).

Point it at the base model and the adapter for a clean fine-tune before/after:

    python code/eval_extraction.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/extract_base.json
    python code/eval_extraction.py --model triage-lora                     --out outputs/extract_lora.json
    python code/eval_extraction.py --compare outputs/extract_base.json outputs/extract_lora.json

Needs a vLLM server on TRIAGE_BASE_URL (default http://localhost:8000/v1).

Metrics
-------
row_exact_match     fraction of rows where pred slot-dict == gold slot-dict
key P / R / F1      did the model emit the right SET of slot keys?
                     FP key = hallucinated slot (the scenario-5 failure mode)
                     FN key = missed slot
value_accuracy      of the keys present in both, fraction with the right value
                     (enum_set compared as a set; numbers with tolerance)
per-slot table      key-level TP/FP/FN + value errors, worst F1 first
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import slots as S  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "..", "data", "eval", "extraction_golden.jsonl")


def load_golden(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def value_matches(slot_id: str, gold, pred) -> bool:
    slot = S.SLOTS_BY_ID.get(slot_id)
    if slot and slot.type == S.ENUM_SET:
        return set(gold or []) == set(pred or [])
    if slot and slot.type == S.INT:
        return _num(gold) == _num(pred)               # severity 7 vs 8 IS different
    if slot and slot.type == S.FLOAT:
        g, p = _num(gold), _num(pred)
        if g is None or p is None:
            return g == p
        return abs(g - p) <= max(1.0, 0.25 * abs(g))  # onset "6h" vs "5h" is fine
    return gold == pred


def score_row(gold: dict, pred: dict) -> dict:
    gk, pk = set(gold), set(pred)
    tp_keys = gk & pk
    fp_keys = pk - gk          # hallucinated
    fn_keys = gk - pk          # missed
    val_ok = {k for k in tp_keys if value_matches(k, gold[k], pred[k])}
    val_bad = tp_keys - val_ok
    return {
        "exact": gold == pred,
        "tp_keys": sorted(tp_keys),
        "fp_keys": sorted(fp_keys),
        "fn_keys": sorted(fn_keys),
        "val_ok": sorted(val_ok),
        "val_bad": sorted(val_bad),
    }


def run(model: str, hint_asked: bool, concurrency: int) -> dict:
    from llm_client import LLMClient

    client = LLMClient(model=model, llm_questions=False, llm_final=False)
    rows = load_golden(_GOLDEN)

    def one(r: dict) -> dict:
        asked = list(r["gold_slots"]) if hint_asked else None
        res = client.extract_slots(
            r["patient_text"], known_slots=r.get("known_slots") or None,
            asked_slots=asked,
        )
        sc = score_row(r["gold_slots"], res.data or {})
        sc.update(tag=r.get("tag", ""), patient_text=r["patient_text"],
                  gold=r["gold_slots"], pred=res.data or {}, ok=res.ok)
        return sc

    out: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(one, r): i for i, r in enumerate(rows)}
        done = 0
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}")

    return aggregate(model, out)


def aggregate(model: str, rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(len(r["tp_keys"]) for r in rows)
    fp = sum(len(r["fp_keys"]) for r in rows)
    fn = sum(len(r["fn_keys"]) for r in rows)
    val_ok = sum(len(r["val_ok"]) for r in rows)
    val_bad = sum(len(r["val_bad"]) for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "val_bad": 0})
    for r in rows:
        for k in r["tp_keys"]:
            per[k]["tp"] += 1
        for k in r["fp_keys"]:
            per[k]["fp"] += 1
        for k in r["fn_keys"]:
            per[k]["fn"] += 1
        for k in r["val_bad"]:
            per[k]["val_bad"] += 1
    per_slot = {}
    for k, d in per.items():
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        rr = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        per_slot[k] = {**d, "precision": round(p, 3), "recall": round(rr, 3),
                       "f1": round(2 * p * rr / (p + rr), 3) if p + rr else 0.0}

    halluc = [{"tag": r["tag"], "patient_text": r["patient_text"],
               "fp_keys": r["fp_keys"], "gold": r["gold"], "pred": r["pred"]}
              for r in rows if r["fp_keys"]]

    return {
        "model": model,
        "n_rows": n,
        "row_exact_match": round(sum(r["exact"] for r in rows) / n, 4),
        "key_precision": round(prec, 4),
        "key_recall": round(rec, 4),
        "key_f1": round(f1, 4),
        "value_accuracy": round(val_ok / (val_ok + val_bad), 4) if val_ok + val_bad else None,
        "counts": {"tp_keys": tp, "fp_keys_hallucinated": fp, "fn_keys_missed": fn,
                   "value_wrong": val_bad},
        "hallucination_rows": halluc,
        "per_slot": dict(sorted(per_slot.items(), key=lambda kv: kv[1]["f1"])),
        "rows": rows,
    }


def report(agg: dict) -> None:
    print(f"\n=== extraction eval -- {agg['model']} ===")
    for k in ("n_rows", "row_exact_match", "key_precision", "key_recall",
              "key_f1", "value_accuracy"):
        print(f"  {k:20} {agg[k]}")
    print(f"  counts               {agg['counts']}")
    if agg["hallucination_rows"]:
        print(f"\n  hallucinated a slot on {len(agg['hallucination_rows'])} row(s):")
        for h in agg["hallucination_rows"][:12]:
            print(f"    +{h['fp_keys']}  {h['patient_text'][:70]!r}")
    print("\n  worst slots by key-F1:")
    for k, d in list(agg["per_slot"].items())[:12]:
        print(f"    {k:32} f1={d['f1']:.2f}  tp={d['tp']} fp={d['fp']} fn={d['fn']} "
              f"val_bad={d['val_bad']}")


def compare(a_path: str, b_path: str) -> None:
    a = json.load(open(a_path)); b = json.load(open(b_path))
    print(f"\n{'metric':22}{a['model'][:24]:>26}{b['model'][:24]:>26}")
    for k in ("row_exact_match", "key_precision", "key_recall", "key_f1",
              "value_accuracy"):
        print(f"{k:22}{a[k]!s:>26}{b[k]!s:>26}")
    print(f"{'hallucinated rows':22}{len(a['hallucination_rows'])!s:>26}"
          f"{len(b['hallucination_rows'])!s:>26}")
    pa, pb = a["per_slot"], b["per_slot"]
    moved = []
    for k in set(pa) | set(pb):
        fa = pa.get(k, {}).get("f1", 0.0); fb = pb.get(k, {}).get("f1", 0.0)
        if abs(fa - fb) >= 0.05:
            moved.append((fb - fa, k, fa, fb))
    if moved:
        print("\n  per-slot key-F1 changes (>=0.05):")
        for d, k, fa, fb in sorted(moved, reverse=True):
            print(f"    {('+' if d > 0 else '')}{d:.2f}  {k:32} {fa:.2f} -> {fb:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("TRIAGE_MODEL",
                                                      "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--hint-asked", action="store_true",
                    help="tell the extractor which slots were 'asked' (= the gold "
                         "keys). Off by default -- pure extraction from text.")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    print(f"extraction eval: {args.model} (hint_asked={args.hint_asked})")
    agg = run(args.model, args.hint_asked, args.concurrency)
    report(agg)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(agg, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
