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
    engine = Engine()
    engine.try_load_index()
    engine_lock = threading.Lock()

    sock_path = config.SOCKET_PATH
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o666)
    srv.listen(16)
    log.info("server ready on %s (index=%s)", sock_path, engine.index is not None)

    def handle(conn: socket.socket) -> None:
        try:
            req = recv_json(conn)
            op = req.get("op")
            if op == "ping":
                send_json(conn, {"ok": True, "indexed": engine.index is not None,
                                 "chunks": len(engine.index.chunks) if engine.index else 0})
                return
            if op == "index":
                with engine_lock:
                    info = engine.build_index(Path(req["corpus"]), req.get("budget"))
                send_json(conn, {"ok": True, **info})
                return
            if op == "query":
                t0 = time.monotonic()
                with engine_lock:
                    if engine.index is None:
                        if not engine.try_load_index():
                            corpus = req.get("corpus")
                            if corpus and Path(corpus).is_dir():
                                engine.build_index(Path(corpus), req.get("index_budget", 200))
                    res = engine.answer(req["query"], req.get("budget"))
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
