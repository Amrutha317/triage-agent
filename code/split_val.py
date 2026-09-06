"""
split_val.py -- deterministic template-level split of the extraction golden set.

data/eval/extraction_golden.jsonl (112 rows, ~97 `tag` templates) is the
canonical full set and stays untouched. This writes two disjoint views of it:

  data/eval/extraction_val.jsonl   -- for model selection (which adapter is best)
  data/eval/extraction_test.jsonl  -- held-out, reported in the writeup

The split is by `tag`, not by row: every row of a template lands entirely in
one side, so near-duplicate phrasings of the same template can't straddle the
split. Tags are sorted and every 5th one goes to val (~20%), so the split is
reproducible with no RNG.

Run:  python code/split_val.py
Then: python code/eval_extraction.py --rows data/eval/extraction_test.jsonl ...
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "..", "data", "eval", "extraction_golden.jsonl")
_VAL = os.path.join(_HERE, "..", "data", "eval", "extraction_val.jsonl")
_TEST = os.path.join(_HERE, "..", "data", "eval", "extraction_test.jsonl")

_EVERY = 5   # every 5th sorted tag -> validation


def main() -> None:
    rows = [json.loads(l) for l in open(_GOLDEN, encoding="utf-8") if l.strip()]
    tags = sorted({r["tag"] for r in rows})
    val_tags = set(tags[::_EVERY])

    val = [r for r in rows if r["tag"] in val_tags]
    test = [r for r in rows if r["tag"] not in val_tags]
    assert len(val) + len(test) == len(rows)
    assert not ({r["tag"] for r in val} & {r["tag"] for r in test}), "tag straddles split"

    for path, part in ((_VAL, val), (_TEST, test)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")

    print(f"{len(rows)} rows / {len(tags)} tags  ->")
    print(f"  val : {len(val):3} rows / {len(val_tags):2} tags  -> {os.path.relpath(_VAL, _HERE + '/..')}")
    print(f"  test: {len(test):3} rows / {len(tags) - len(val_tags):2} tags  -> {os.path.relpath(_TEST, _HERE + '/..')}")
    print("  val tags:", ", ".join(sorted(val_tags)))


if __name__ == "__main__":
    main()
