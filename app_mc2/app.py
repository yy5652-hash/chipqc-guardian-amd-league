#!/usr/bin/env python3
"""Thin client for the AMD Mini-Challenge 2 contract:

    python3 /app/app.py --input-image /app/input/image_01.png
    -> /app/output/image_01_output.json  {"text": "...", "confidence": 0.94}

Always writes the output file before the 30 s per-image budget; the model lives in server.py.
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
OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))


def rpc(req: dict, timeout: float):
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
        os.kill(int(Path(config.PID_PATH).read_text().strip()), 0)
        return True
    except Exception:
        return False


def ensure_server(wait_s: float):
    end = time.monotonic() + wait_s
    spawned = False
    while time.monotonic() < end:
        r = rpc({"op": "ping"}, timeout=2)
        if r and r.get("ok"):
            return r
        if not spawned and not server_alive():
            logf = open(config.LOG_PATH, "ab")
            subprocess.Popen([sys.executable, str(SERVER)], stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True, cwd=str(HERE))
            spawned = True
        time.sleep(0.5)
    return None


def write_output(stem: str, text: str, confidence: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{stem}_output.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"text": text if isinstance(text, str) else "", "confidence": float(confidence)},
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.monotonic()
    stem = args.input_image.stem
    text, conf = "", 0.0
    try:
        deadline = config.CLIENT_DEADLINE_S
        r = ensure_server(wait_s=max(1.0, deadline - 6 - (time.monotonic() - t0)))
        if r:
            remaining = deadline - (time.monotonic() - t0)
            if remaining > 2:
                res = rpc({"op": "ocr", "path": str(args.input_image.resolve()), "budget": max(3.0, min(20.0, remaining - 4))},
                          timeout=remaining)
                if res and isinstance(res.get("text"), str):
                    text, conf = res["text"], float(res.get("confidence", 0.5))
    except Exception as e:  # noqa: BLE001
        print(f"ocr failed: {e}", file=sys.stderr)
    write_output(stem, text, conf)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"fatal: {e}", file=sys.stderr)
        try:
            argv = sys.argv
            if "--input-image" in argv:
                write_output(Path(argv[argv.index("--input-image") + 1]).stem, "", 0.0)
        except Exception:
            pass
        sys.exit(0)
