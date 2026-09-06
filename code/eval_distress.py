"""
eval_distress.py -- measure classify_distress in isolation.

Derives a labeled set from data/eval/scenarios.jsonl: each scenario's
patient_sim chief-complaint utterance (the open-ended first turn) + the
observed-flag values in its `facts`. Runs classify_distress() on the text and
scores per-flag precision / recall / F1.

This isolates the distress classifier from the end-to-end pipeline bug where a
later calm turn overwrites a turn-1 escalation -- here every utterance is
scored once, so the number reflects the model, not the merge logic.

    python code/eval_distress.py --model meta-llama/Llama-3.1-8B-Instruct --out outputs/distress_base.json
    python code/eval_distress.py --model triage-lora                     --out outputs/distress_lora.json
    python code/eval_distress.py --compare outputs/distress_base.json outputs/distress_lora.json

Needs a vLLM server on TRIAGE_BASE_URL. `--dump PATH` writes the derived
labeled set as jsonl for inspection.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import slots as S  # noqa: E402
from patient_sim import make_patient_sim  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCENARIOS = os.path.join(_HERE, "..", "data", "eval", "scenarios.jsonl")

# the six observed_only slots classify_distress is responsible for
FLAGS = [s.id for s in S.ALL_SLOTS if s.observed_only]


def labeled_set() -> list[dict]:
    out = []
    with open(_SCENARIOS, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            sc = json.loads(ln)
            facts = sc["facts"]
            out.append({
                "name": sc["name"],
                "text": make_patient_sim(facts)(None),
                "gold": {f: bool(facts.get(f, False)) for f in FLAGS},
            })
    return out


def run(model: str, concurrency: int) -> dict:
    from llm_client import LLMClient

    client = LLMClient(model=model, llm_questions=False, llm_final=False)
    data = labeled_set()

    def one(row: dict) -> dict:
        res = client.classify_distress(row["text"])
        pred = {f: bool((res.data or {}).get(f, False)) for f in FLAGS}
        return {**row, "pred": pred, "rationale": res.meta.get("rationale", ""),
                "ok": res.ok}

    out: list[dict] = [None] * len(data)  # type: ignore[list-item]
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(one, r): i for i, r in enumerate(data)}
        done = 0
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 15 == 0 or done == len(data):
                print(f"  {done}/{len(data)}")
    return aggregate(model, out)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def aggregate(model: str, rows: list[dict]) -> dict:
    per = {}
    TP = FP = FN = 0
    for f in FLAGS:
        tp = sum(1 for r in rows if r["gold"][f] and r["pred"][f])
        fp = sum(1 for r in rows if not r["gold"][f] and r["pred"][f])
        fn = sum(1 for r in rows if r["gold"][f] and not r["pred"][f])
        tn = sum(1 for r in rows if not r["gold"][f] and not r["pred"][f])
        p, r, f1 = _prf(tp, fp, fn)
        per[f] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                  "precision": p, "recall": r, "f1": f1}
        TP += tp; FP += fp; FN += fn

    mp, mr, mf = _prf(TP, FP, FN)

    # scenario-level: did we raise EVERY flag the scenario needs, and no
    # spurious escalation flag? (life_threatening / very_sick are the ones that
    # actually change the disposition tier)
    exact = sum(1 for r in rows if r["gold"] == r["pred"])
    missed_needed = [r["name"] for r in rows
                     if any(r["gold"][f] and not r["pred"][f] for f in FLAGS)]
    spurious = [r["name"] for r in rows
                if any(not r["gold"][f] and r["pred"][f] for f in FLAGS)]

    return {
        "model": model,
        "n": len(rows),
        "flag_exact_match": round(exact / len(rows), 4),
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "per_flag": per,
        "scenarios_missing_a_needed_flag": missed_needed,
        "scenarios_with_a_spurious_flag": spurious,
        "rows": rows,
    }


def report(a: dict) -> None:
    print(f"\n=== distress eval -- {a['model']} ===")
    for k in ("n", "flag_exact_match", "micro_precision", "micro_recall", "micro_f1"):
        print(f"  {k:22} {a[k]}")
    print(f"\n  {'flag':34}{'supp':>5}{'P':>7}{'R':>7}{'F1':>7}   tp/fp/fn")
    for f, d in a["per_flag"].items():
        print(f"  {f:34}{d['support']:>5}{d['precision']:>7}{d['recall']:>7}"
              f"{d['f1']:>7}   {d['tp']}/{d['fp']}/{d['fn']}")
    if a["scenarios_missing_a_needed_flag"]:
        print(f"\n  MISSED a needed flag ({len(a['scenarios_missing_a_needed_flag'])}):")
        for n in a["scenarios_missing_a_needed_flag"]:
            print(f"    {n}")
    if a["scenarios_with_a_spurious_flag"]:
        print(f"\n  spurious flag ({len(a['scenarios_with_a_spurious_flag'])}):")
        for n in a["scenarios_with_a_spurious_flag"]:
            print(f"    {n}")


def compare(a_path: str, b_path: str) -> None:
    a = json.load(open(a_path)); b = json.load(open(b_path))
    print(f"\n{'metric':22}{a['model'][:24]:>26}{b['model'][:24]:>26}")
    for k in ("flag_exact_match", "micro_precision", "micro_recall", "micro_f1"):
        print(f"{k:22}{a[k]!s:>26}{b[k]!s:>26}")
    print(f"{'missed needed flag':22}"
          f"{len(a['scenarios_missing_a_needed_flag'])!s:>26}"
          f"{len(b['scenarios_missing_a_needed_flag'])!s:>26}")
    print(f"{'spurious flag':22}{len(a['scenarios_with_a_spurious_flag'])!s:>26}"
          f"{len(b['scenarios_with_a_spurious_flag'])!s:>26}")
    print("\n  per-flag F1:")
    for f in a["per_flag"]:
        fa = a["per_flag"][f]["f1"]; fb = b["per_flag"][f]["f1"]
        mark = "" if abs(fa - fb) < 0.05 else ("   <-- " + ("up" if fb > fa else "down"))
        print(f"    {f:34} {fa:.2f} -> {fb:.2f}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("TRIAGE_MODEL",
                                                      "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dump", default=None, help="write the derived labeled set here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for r in labeled_set():
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {args.dump}")
        return

    print(f"distress eval: {args.model}  ({len(FLAGS)} flags)")
    agg = run(args.model, args.concurrency)
    report(agg)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(agg, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
