#!/usr/bin/env python3
"""In-process end-to-end run against a corpus + sample-questions.json, scored like the grader.

  python3 tests/run_local.py --corpus mc3-corpus --questions sample-questions.json [--stub] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated question numbers")
    args = ap.parse_args()
    if args.stub and os.environ.get("MC3_STUB") not in {"1", "oracle"}:
        os.environ["MC3_STUB"] = "1"
    if args.device:
        os.environ["MC3_DEVICE"] = args.device
    os.environ["MC3_INDEX_DIR"] = args.index_dir or str(HERE.parent / ".index-local")
    os.environ.setdefault("MC3_LOG", str(HERE.parent / ".index-local" / "server.log"))
    Path(os.environ["MC3_INDEX_DIR"]).mkdir(parents=True, exist_ok=True)

    from mc3.pipeline import Engine
    from mc3.util import normalize_answer, setup_logging

    setup_logging("mc3")
    eng = Engine()
    t0 = time.monotonic()
    info = eng.build_index(args.corpus.resolve())
    print(f"\nINDEX: {json.dumps(info, ensure_ascii=False)[:1500]}  ({time.monotonic() - t0:.1f}s)\n")

    qs = json.loads(args.questions.read_text())["queries"]
    only = {int(x) for x in args.only.split(",")} if args.only else None
    score = 0
    rows = []
    for q in qs:
        if only and q["n"] not in only:
            continue
        t1 = time.monotonic()
        res = eng.answer(q["query"])
        dt = time.monotonic() - t1
        exp_ans = {normalize_answer(q["expected_answer"])} | {normalize_answer(a) for a in q.get("answer_aliases", [])}
        got = normalize_answer(res["answer"])
        ans_ok = got in exp_ans
        cit_ok = sorted(res["citations"]) == sorted(q["expected_citations"])
        ok = ans_ok and cit_ok
        score += 20 if ok else 0
        rows.append((q["n"], ok, ans_ok, cit_ok, res["answer"], q["expected_answer"], res["citations"],
                     q["expected_citations"], dt))
    print(f"{'#':>2} {'ok':>3} {'ans':>3} {'cit':>3} {'secs':>5}  got -> expected | citations got -> expected")
    for n, ok, a, c, got, exp, cg, ce, dt in rows:
        print(f"{n:>2} {str(ok):>3} {str(a):>5} {str(c):>5} {dt:5.1f}  {got!r} -> {exp!r} | {cg} -> {ce}")
    print(f"\nSCORE {score}/{20 * len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
