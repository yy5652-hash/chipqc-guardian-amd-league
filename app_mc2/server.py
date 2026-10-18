#!/usr/bin/env python3
"""Resident Qwen3-VL OCR server behind a unix socket (Match 2)."""
from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mc3 import config  # noqa: E402
from mc3.util import setup_logging  # noqa: E402

log = setup_logging("mc2.server")


def recv_json(conn):
    buf = b""
    while not buf.endswith(b"\n"):
        part = conn.recv(65536)
        if not part:
            break
        buf += part
    return json.loads(buf.decode("utf-8")) if buf.strip() else {}


def send_json(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def main() -> int:
    lock = open("/tmp/mc3-server.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("another server holds the lock; exiting")
        return 0
    Path(config.PID_PATH).write_text(str(os.getpid()))

    state = {"ocr": None, "error": None}
    ready = threading.Event()
    lock2 = threading.Lock()

    def load():
        try:
            if config.STUB:
                from mc3.llm import StubVLM
                vlm = StubVLM()
            else:
                from mc3.llm import VLM
                vlm = VLM(config.VLM_PATH, config.DEVICE, config.DTYPE)
                if config.DEVICE != "cpu":
                    try:
                        import torch
                        torch.cuda.synchronize()
                        log.info("GPU memory allocated after load: %.2f GiB", torch.cuda.memory_allocated() / 2**30)
                    except Exception as e:  # noqa: BLE001
                        log.warning("cuda check: %s", e)
                vlm.warmup()
            from ocr import OCR
            state["ocr"] = OCR(vlm)
        except Exception as e:  # noqa: BLE001
            state["error"] = str(e)
            log.error("model load failed: %s\n%s", e, traceback.format_exc())
        finally:
            ready.set()
            log.info("OCR engine ready=%s", state["ocr"] is not None)

    threading.Thread(target=load, daemon=True).start()

    sock_path = config.SOCKET_PATH
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o666)
    srv.listen(16)
    log.info("OCR server socket open on %s (model loading in background)", sock_path)

    def handle(conn):
        try:
            req = recv_json(conn)
            if req.get("op") == "ping":
                send_json(conn, {"ok": True, "ready": ready.is_set() and state["ocr"] is not None, "error": state["error"]})
                return
            if req.get("op") == "ocr":
                deadline_at = float(req.get("deadline_at") or (time.time() + float(req.get("budget", 20.0))))
                if not ready.wait(max(0.0, deadline_at - time.time() - 3)) or state["ocr"] is None:
                    send_json(conn, {"ok": False, "text": "", "confidence": 0.0, "note": "model not ready"})
                    return
                with lock2:
                    budget = deadline_at - time.time()
                    if budget < 3.0:
                        send_json(conn, {"ok": False, "text": "", "confidence": 0.0, "note": "no time left"})
                        return
                    res = state["ocr"].read(req["path"], budget)
                send_json(conn, {"ok": True, **res})
                return
            send_json(conn, {"ok": False, "error": "unknown op"})
        except Exception as e:  # noqa: BLE001
            log.error("request failed: %s\n%s", e, traceback.format_exc())
            try:
                send_json(conn, {"ok": False, "error": str(e), "text": "", "confidence": 0.0})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        log.error("server crashed: %s\n%s", e, traceback.format_exc())
        sys.exit(1)
