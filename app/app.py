#!/usr/bin/env python3
"""Thin client for the AMD Mini-Challenge 3 contract.

  python3 /app/app.py --index /app/corpus
  python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

All model work lives in server.py (resident behind a unix socket). This script
never takes longer than the per-question budget: if the server does not answer
in time it writes a refusal, which costs one question instead of the whole run.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mc3 import config  # noqa: E402

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server.py"


def rpc(req: dict, timeout: float) -> dict | None:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(config.SOCKET_PATH)
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            part = s.recv(65536)
            if not part:
                break
            buf += part
        s.close()
        return json.loads(buf.decode("utf-8")) if buf.strip() else None
    except Exception:
        return None


def server_alive() -> bool:
    try:
        pid = int(Path(config.PID_PATH).read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def spawn_server() -> None:
    logf = open(config.LOG_PATH, "ab")
    subprocess.Popen([sys.executable, str(SERVER)], stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True, close_fds=True, cwd=str(HERE))


def ensure_server(wait_s: float) -> dict | None:
    end = time.monotonic() + wait_s
    spawned = False
    while time.monotonic() < end:
        r = rpc({"op": "ping"}, timeout=2)
        if r and r.get("ok"):
            return r
        if not spawned and not server_alive():
            spawn_server()
            spawned = True
        time.sleep(0.5)
    return None


def write_output(query_id: str, answer: str, citations: list[str], confidence: float) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"{query_id}_output.json"
    tmp = out.with_suffix(".json.tmp")
    payload = {"answer": answer if isinstance(answer, str) else "", "citations": [str(c) for c in citations],
               "confidence": float(confidence)}
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()
    t0 = time.monotonic()

    if args.index is not None:
        r = ensure_server(wait_s=config.INDEX_BUDGET_S + 120)
        if not r:
            print("server did not come up; index skipped", file=sys.stderr)
            return 0
        remaining = max(30.0, config.INDEX_BUDGET_S + 120 - (time.monotonic() - t0))
        res = rpc({"op": "index", "corpus": str(args.index.resolve()), "budget": config.INDEX_BUDGET_S},
                  timeout=remaining)
        print(json.dumps(res or {"ok": False}, ensure_ascii=False)[:2000])
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    qid = args.query_id
    answer, citations, conf = "", [], 0.0
    try:
        deadline = config.CLIENT_DEADLINE_S
        r = ensure_server(wait_s=max(1.0, deadline - 6 - (time.monotonic() - t0)))
        if r:
            remaining = deadline - (time.monotonic() - t0)
            if remaining > 2:
                res = rpc({"op": "query", "query": args.query, "corpus": str(args.corpus),
                           "budget": max(3.0, min(config.SERVER_SOFT_BUDGET_S, remaining - 4)),
                           "index_budget": max(5.0, remaining - 8)}, timeout=remaining)
                if res and isinstance(res.get("answer"), str) and isinstance(res.get("citations"), list):
                    answer, citations, conf = res["answer"], res["citations"], res.get("confidence", 0.5)
    except Exception as e:  # noqa: BLE001
        print(f"query failed: {e}", file=sys.stderr)
    write_output(qid, answer, citations, conf)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - never leave without an output file
        print(f"fatal: {e}", file=sys.stderr)
        try:
            qid = None
            argv = sys.argv
            if "--query-id" in argv:
                qid = argv[argv.index("--query-id") + 1]
            if qid:
                write_output(qid, "", [], 0.0)
        except Exception:
            pass
        sys.exit(0)
