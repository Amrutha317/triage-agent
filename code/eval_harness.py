"""
eval_harness.py -- run every scenario in data/eval/scenarios.jsonl through a
full simulated conversation and score the result against gold.

Two modes:
  --offline   OracleLLMClient (perfect extraction, no network) -- validates
              the whole pipeline mechanically; not a measure of model quality.
  (default)   the real LLMClient against TRIAGE_BASE_URL / TRIAGE_MODEL --
              needs the pod. Point TRIAGE_MODEL at "triage-lora" to eval the
              fine-tuned adapter with the exact same harness for baseline vs
              LoRA comparison.

Metrics:
  triage accuracy         exact disposition match vs gold
  workflow accuracy       exact (disposition, rule_id) match -- "right answer,
                           right reason", not a coincidence
  under/over-triage rate  fraction landing less/more urgent than gold, by tier
  red-flag recall         of gold==CALL_EMS_911_NOW cases, fraction predicted there
  red-flag recall (broad) of gold in {911, ED-now} cases, fraction predicted in
                           that same top-2 set (didn't fall below ED-now)
  confusion matrix        gold -> predicted disposition counts
  guardrail trigger rate  how often a rendered message needed the safety fallback
  latency                 TTFT / total, per call type, p50 / p90

Run:
  python code/eval_harness.py --offline
  python code/eval_harness.py --scenarios data/eval/scenarios.jsonl --out outputs/eval_offline.json
  TRIAGE_MODEL=triage-lora python code/eval_harness.py --out outputs/eval_lora.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from agent import TriageSession  # noqa: E402
from decision_engine import RulesEngine  # noqa: E402
from patient_sim import make_patient_sim  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SCENARIOS = os.path.join(_HERE, "..", "data", "eval", "scenarios.jsonl")

ENGINE = RulesEngine()
TIER_RANK = {name: i for i, name in enumerate(ENGINE.order)}   # 0 = most urgent
_TOP2 = {"CALL_EMS_911_NOW", "GO_TO_ED_NOW"}


def load_scenarios(path: str) -> list[dict]:
    scenarios = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


def run_scenario(make_client, scenario: dict, max_turns: int = 30) -> dict:
    facts = scenario["facts"]
    client = make_client(facts)
    session = TriageSession(llm_client=client)

    t0 = time.perf_counter()
    result = session.run_conversation(make_patient_sim(facts), max_turns=max_turns)
    wall = time.perf_counter() - t0

    ttfts, totals = [], []
    guardrail_hits = 0
    for t in result.turns:
        for v in (t.extract_ttft, t.distress_ttft):
            if v is not None:
                ttfts.append(v)
        for v in (t.extract_total, t.distress_total):
            if v is not None:
                totals.append(v)
        if t.guardrail_triggered:
            guardrail_hits += 1

    return {
        "name": scenario["name"],
        "gold_disposition": scenario["gold_disposition"],
        "gold_rule_id": scenario["gold_rule_id"],
        "pred_disposition": result.disposition,
        "pred_rule_id": result.rule_id,
        "done": result.done,
        "n_turns": result.n_turns,
        "wall_seconds": wall,
        "call_ttft_seconds": ttfts,
        "call_total_seconds": totals,
        "guardrail_hits": guardrail_hits,
    }


def _pct(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "p50": None, "p90": None}
    xs = sorted(xs)
    return {
        "mean": statistics.fmean(xs),
        "p50": statistics.median(xs),
        "p90": xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))],
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    triage_correct = sum(1 for r in rows if r["pred_disposition"] == r["gold_disposition"])
    workflow_correct = sum(
        1 for r in rows
        if r["pred_disposition"] == r["gold_disposition"] and r["pred_rule_id"] == r["gold_rule_id"]
    )
    undone = sum(1 for r in rows if not r["done"])

    # per-row tier delta (>0 = under-triage / less urgent than gold), kept
    # aligned to its row -- some rows have no delta (never terminated, or an
    # unrecognized disposition), so this must NOT be a separately-zipped list
    scored = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        gold, pred = r["gold_disposition"], r["pred_disposition"]
        confusion[gold][pred or "NONE"] += 1
        if pred is not None and gold in TIER_RANK and pred in TIER_RANK:
            scored.append((r, TIER_RANK[pred] - TIER_RANK[gold]))

    deltas = [d for _, d in scored]
    under = sum(1 for d in deltas if d > 0)
    over = sum(1 for d in deltas if d < 0)

    redflag_rows = [r for r in rows if r["gold_disposition"] == "CALL_EMS_911_NOW"]
    redflag_recall = (
        sum(1 for r in redflag_rows if r["pred_disposition"] == "CALL_EMS_911_NOW") / len(redflag_rows)
        if redflag_rows else None
    )
    top2_rows = [r for r in rows if r["gold_disposition"] in _TOP2]
    redflag_recall_broad = (
        sum(1 for r in top2_rows if r["pred_disposition"] in _TOP2) / len(top2_rows)
        if top2_rows else None
    )

    all_ttft = [x for r in rows for x in r["call_ttft_seconds"]]
    all_total = [x for r in rows for x in r["call_total_seconds"]]
    wall_seconds = [r["wall_seconds"] for r in rows]
    n_turns = [r["n_turns"] for r in rows]
    guardrail_rate = sum(r["guardrail_hits"] for r in rows) / sum(max(r["n_turns"], 1) for r in rows)

    return {
        "n_scenarios": len(rows),
        "triage_accuracy": triage_correct / n,
        "workflow_accuracy": workflow_correct / n,
        "did_not_terminate": undone,
        "under_triage_rate": under / len(deltas) if deltas else None,
        "over_triage_rate": over / len(deltas) if deltas else None,
        "under_triage_examples": [
            {"name": r["name"], "gold": r["gold_disposition"], "pred": r["pred_disposition"]}
            for r, d in scored if d > 0
        ][:10],
        "red_flag_recall_911": redflag_recall,
        "red_flag_recall_top2_tier": redflag_recall_broad,
        "guardrail_trigger_rate": guardrail_rate,
        "n_turns": _pct(n_turns),
        "wall_seconds_per_scenario": _pct(wall_seconds),
        "call_ttft_seconds": _pct(all_ttft),
        "call_total_seconds": _pct(all_total),
        "confusion_matrix": {g: dict(p) for g, p in confusion.items()},
    }


def format_report(agg: dict, model_label: str = "") -> str:
    L = [f"=== eval report {('- ' + model_label) if model_label else ''} ==="]
    for k in ("n_scenarios", "triage_accuracy", "workflow_accuracy", "did_not_terminate",
              "under_triage_rate", "over_triage_rate", "red_flag_recall_911",
              "red_flag_recall_top2_tier", "guardrail_trigger_rate"):
        v = agg.get(k)
        L.append(f"  {k:28} {v:.3f}" if isinstance(v, float) else f"  {k:28} {v}")
    nt = agg["n_turns"]
    L.append(f"  n_turns                     mean={nt['mean']:.1f} p50={nt['p50']:.0f} p90={nt['p90']:.0f}")
    ws = agg["wall_seconds_per_scenario"]
    if ws["mean"] is not None:
        L.append(f"  wall_seconds/scenario       mean={ws['mean']:.2f} p50={ws['p50']:.2f} p90={ws['p90']:.2f}")
    ttft = agg["call_ttft_seconds"]
    if ttft["mean"] is not None:
        L.append(f"  call TTFT (s)               mean={ttft['mean']:.3f} p50={ttft['p50']:.3f} p90={ttft['p90']:.3f}")
    total = agg["call_total_seconds"]
    if total["mean"] is not None:
        L.append(f"  call total (s)              mean={total['mean']:.3f} p50={total['p50']:.3f} p90={total['p90']:.3f}")
    if agg["under_triage_examples"]:
        L.append("  under-triage examples (most safety-relevant to review first):")
        for ex in agg["under_triage_examples"]:
            L.append(f"    {ex['name']:50} gold={ex['gold']:28} pred={ex['pred']}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", default=_DEFAULT_SCENARIOS)
    ap.add_argument("--offline", action="store_true",
                    help="use OracleLLMClient (perfect extraction, no pod) to "
                         "validate the pipeline mechanically")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    ap.add_argument("--label", default=None, help="model label for the printed report")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="scenarios to run in parallel (vLLM batches these "
                         "server-side; this was previously fully sequential -- "
                         "one full 8-turn conversation blocking the next). "
                         "1 = old sequential behavior.")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.offline:
        from oracle_client import OracleLLMClient
        make_client = lambda facts: OracleLLMClient(facts)  # noqa: E731
        label = args.label or "offline (oracle, no LLM)"
    else:
        from llm_client import LLMClient
        shared = LLMClient()
        make_client = lambda facts: shared  # noqa: E731
        label = args.label or shared.model

    print(f"running {len(scenarios)} scenarios ({label}), "
          f"concurrency={args.concurrency}...")
    rows: list[dict] = [None] * len(scenarios)  # type: ignore[list-item]
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        # index-preserving: rows stays in scenario order regardless of which
        # thread finishes first, but the calls themselves run concurrently --
        # the openai client is safe to share across threads (each call is a
        # self-contained sync HTTP round trip with no shared mutable state).
        futures = {ex.submit(run_scenario, make_client, s): i
                  for i, s in enumerate(scenarios)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            rows[i] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(scenarios):
                print(f"  {done}/{len(scenarios)}")

    agg = aggregate(rows)
    print()
    print(format_report(agg, model_label=label))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"label": label, "aggregate": agg, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
