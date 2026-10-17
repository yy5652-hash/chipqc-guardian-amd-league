#!/usr/bin/env python3
"""In-process Match 2 OCR run over the sample images, scored with the grader's normalisation.

  MC3_VLM_PATH=... python3 tests/run_mc2.py --samples external/mc2/samples.json --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app_mc2"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--budget", type=float, default=600.0)
    args = ap.parse_args()
    if args.device:
        os.environ["MC3_DEVICE"] = args.device
    os.environ.setdefault("MC3_LOG", str(HERE.parent / ".index-local" / "mc2.log"))
    Path(os.environ["MC3_LOG"]).parent.mkdir(parents=True, exist_ok=True)
    from mc3 import config
    from mc3.util import setup_logging
    from mc3.llm import VLM
    from ocr import OCR, norm
    setup_logging("mc2")
    vlm = VLM(config.VLM_PATH, config.DEVICE, config.DTYPE)
    ocr = OCR(vlm)
    data = json.loads(args.samples.read_text())["samples"]
    only = {int(x) for x in args.only.split(",")} if args.only else None
    base = args.samples.parent / "samples"
    score = 0
    rows = []
    for s in data:
        if only and s["n"] not in only:
            continue
        t0 = time.monotonic()
        img_path = base / s["file"]
        if not img_path.exists():
            for alt in ("hard", "samples", "."):
                cand = args.samples.parent / alt / s["file"]
                if cand.exists():
                    img_path = cand
                    break
        res = ocr.read(str(img_path), deadline_s=args.budget)
        dt = time.monotonic() - t0
        ok = norm(res["text"]) == norm(s["expected"])
        score += 20 if ok else 0
        rows.append((s["n"], ok, res["text"], s["expected"], res.get("type"), res.get("confidence"), res.get("passes"), dt))
    for n, ok, got, exp, kind, conf, passes, dt in rows:
        print(f"{n:>2} {str(ok):>5} {dt:5.1f}s  {got!r} -> {exp!r}  type={kind} conf={conf} passes={passes}")
    print(f"\nSCORE {score}/{20 * len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
