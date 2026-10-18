#!/usr/bin/env python3
"""Resident model + index server behind a unix socket. Started by the container CMD
(supervise.sh) and, if it is ever missing, by app.py."""
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

log = setup_logging("mc3.server")


def recv_json(conn: socket.socket) -> dict:
    buf = b""
    while not buf.endswith(b"\n"):
        part = conn.recv(65536)
        if not part:
            break
        buf += part
    return json.loads(buf.decode("utf-8")) if buf.strip() else {}


def send_json(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def main() -> int:
    lock = open("/tmp/mc3-server.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("another server holds the lock; exiting")
        return 0
    Path(config.PID_PATH).write_text(str(os.getpid()))

    from mc3.pipeline import Engine

    state = {"engine": None, "error": None}
    ready = threading.Event()
    engine_lock = threading.Lock()

    def load() -> None:
        try:
            eng = Engine()
            try:
                if getattr(eng, "vlm", None) is not None and hasattr(eng.vlm, "warmup"):
                    eng.vlm.warmup()
            except Exception as e:  # noqa: BLE001
                log.warning("warmup: %s", e)
            eng.try_load_index()
            state["engine"] = eng
        except Exception as e:  # noqa: BLE001
            state["error"] = str(e)
            log.error("model load failed: %s\n%s", e, traceback.format_exc())
        finally:
            ready.set()
            log.info("engine ready=%s (load error=%s)", state["engine"] is not None, state["error"])

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
    log.info("server socket open on %s (models loading in background)", sock_path)

    def wait_ready(deadline_at: float | None) -> bool:
        timeout = None if deadline_at is None else max(0.0, deadline_at - time.time())
        return ready.wait(timeout) and state["engine"] is not None

    def handle(conn: socket.socket) -> None:
        try:
            req = recv_json(conn)
            op = req.get("op")
            if op == "ping":
                eng = state["engine"]
                send_json(conn, {"ok": True, "ready": ready.is_set() and eng is not None, "error": state["error"],
                                 "indexed": bool(eng and eng.index is not None),
                                 "chunks": len(eng.index.chunks) if eng and eng.index else 0})
                return
            if op == "index":
                deadline_at = req.get("deadline_at")
                if not wait_ready(deadline_at):
                    send_json(conn, {"ok": False, "error": "models not ready before the index deadline"})
                    return
                with engine_lock:
                    budget = req.get("budget")
                    if deadline_at:
                        budget = max(10.0, min(budget or 1e9, deadline_at - time.time()))
                    info = state["engine"].build_index(Path(req["corpus"]), budget)
                send_json(conn, {"ok": True, **info})
                return
            if op == "query":
                t0 = time.monotonic()
                deadline_at = req.get("deadline_at") or (time.time() + float(req.get("budget") or config.SERVER_SOFT_BUDGET_S))
                if not wait_ready(deadline_at - 4):
                    send_json(conn, {"answer": "", "citations": [], "confidence": 0.0, "note": "models not ready"})
                    return
                with engine_lock:
                    eng = state["engine"]
                    remaining = deadline_at - time.time()
                    if remaining < 4.0:  # an earlier request overran: refuse instead of cascading
                        log.warning("query skipped: only %.1fs left after waiting for the engine", remaining)
                        send_json(conn, {"answer": "", "citations": [], "confidence": 0.0, "note": "no time left"})
                        return
                    if eng.index is None and not eng.try_load_index():
                        corpus = req.get("corpus")
                        if corpus and Path(corpus).is_dir():
                            eng.build_index(Path(corpus), max(5.0, remaining - 8))
                    res = eng.answer(req["query"], max(3.0, deadline_at - time.time()))
                res["seconds"] = round(time.monotonic() - t0, 2)
                send_json(conn, res)
                return
            send_json(conn, {"ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # noqa: BLE001
            log.error("request failed: %s\n%s", e, traceback.format_exc())
            try:
                send_json(conn, {"ok": False, "error": str(e), "answer": "", "citations": [], "confidence": 0.0})
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
