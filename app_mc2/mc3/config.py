from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MC3_MODELS_DIR", "/models"))
VLM_PATH = Path(os.environ.get("MC3_VLM_PATH", str(MODELS_DIR / "Qwen3-VL-8B-Instruct")))
EMB_PATH = Path(os.environ.get("MC3_EMB_PATH", str(MODELS_DIR / "Qwen3-Embedding-0.6B")))

INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", os.environ.get("MC3_OUTPUT_DIR", "/app/output")))
SOCKET_PATH = os.environ.get("MC3_SOCKET", "/tmp/mc3.sock")
PID_PATH = os.environ.get("MC3_PID", "/tmp/mc3-server.pid")
LOG_PATH = os.environ.get("MC3_LOG", "/tmp/mc3-server.log")

STUB = os.environ.get("MC3_STUB", "0") in {"1", "oracle"}          # no models: plumbing tests only
DEVICE = os.environ.get("MC3_DEVICE", "cuda")           # ROCm torch exposes the GPU as "cuda"
DTYPE = os.environ.get("MC3_DTYPE", "bfloat16")

TOP_K = int(os.environ.get("MC3_TOP_K", "12"))
HOP2_K = int(os.environ.get("MC3_HOP2_K", "8"))
MAX_CONTEXT_CHUNKS = int(os.environ.get("MC3_MAX_CTX", "16"))
MAX_FILE_BYTES = int(os.environ.get("MC3_MAX_FILE_BYTES", str(64 * 1024 * 1024)))
MAX_TEXT_CHARS = 1_500_000
MAX_PDF_PAGES = int(os.environ.get("MC3_MAX_PDF_PAGES", "120"))
MAX_OCR_PAGES_PER_PDF = int(os.environ.get("MC3_MAX_OCR_PAGES", "12"))
PARSE_TIMEOUT_S = float(os.environ.get("MC3_PARSE_TIMEOUT_S", "90"))
IMAGE_MAX_SIDE = int(os.environ.get("MC3_IMAGE_MAX_SIDE", "1536"))

# Per-question budget is 30 s wall clock from exec to the JSON file. The client
# writes a refusal at CLIENT_DEADLINE_S no matter what; the server stops doing
# optional work (hop 2, ablations) once SERVER_SOFT_BUDGET_S has elapsed.
CLIENT_DEADLINE_S = float(os.environ.get("MC3_CLIENT_DEADLINE_S", "26"))
SERVER_SOFT_BUDGET_S = float(os.environ.get("MC3_SERVER_SOFT_BUDGET_S", "19"))
INDEX_BUDGET_S = float(os.environ.get("MC3_INDEX_BUDGET_S", "420"))

CHUNK_CHARS = int(os.environ.get("MC3_CHUNK_CHARS", "1400"))
CHUNK_OVERLAP = int(os.environ.get("MC3_CHUNK_OVERLAP", "200"))
ROWS_PER_CHUNK = int(os.environ.get("MC3_ROWS_PER_CHUNK", "8"))

# Written by supervise.sh when the container starts; budgets are anchored to it.
START_FILE = os.environ.get("MC3_START_FILE", "/tmp/mc3-container-start")
STARTUP_BUDGET_S = float(os.environ.get("MC3_STARTUP_BUDGET_S", "600"))
MAX_ANSWER_IMAGES = int(os.environ.get("MC3_MAX_ANSWER_IMAGES", "2"))
ANSWER_IMAGE_MAX_SIDE = int(os.environ.get("MC3_ANSWER_IMAGE_MAX_SIDE", "1024"))


def seconds_since_container_start() -> float | None:
    import time as _t
    try:
        return _t.time() - float(open(START_FILE).read().strip())
    except Exception:
        return None
