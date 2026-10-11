#!/bin/bash
# Run the Match 3 pipeline end-to-end inside an AMD hackathon JupyterLab session
# (same base image as the submission, real GPU, network available).
#
#   1. upload mc3-kit.tgz (made by notebook/make_kit.sh) to the session and run:
#        tar xzf mc3-kit.tgz && bash mc3-kit/notebook/run_on_amd.sh
#   2. results land in mc3-kit/results/ (sample score table + timings + VRAM peak)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PERSIST=/persistent; [ -d /workspace ] && [ ! -d /persistent ] && PERSIST=/workspace
MODELS="$PERSIST/mc3-models"; mkdir -p "$MODELS" "$ROOT/results"
export MC3_MODELS_DIR="$MODELS" MC3_INDEX_DIR="$ROOT/.index" MC3_LOG="$ROOT/results/server.log" HF_HUB_DISABLE_TELEMETRY=1

echo "== torch"; python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo "== deps (no-deps install, same list as the Dockerfile)"
pip install --no-cache-dir --no-deps -q -r "$ROOT/app/requirements.txt"
python3 -c "import torch; assert '+rocm' in torch.__version__, torch.__version__; import transformers, fitz, docx, openpyxl; print('deps ok', transformers.__version__)"

echo "== weights (cached under $MODELS)"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-8B-Instruct"):
    dest = os.path.join("$MODELS", repo.split("/")[1])
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=["*.json","*.safetensors","*.txt","merges.txt","vocab.json","tokenizer*"])
    print("ok", dest)
PY

echo "== sample corpus"
KIT="$ROOT/sample"; mkdir -p "$KIT/mc3-corpus/archive"; chmod 000 "$KIT/mc3-corpus/vendor/internal_audit.txt" || true

echo "== VRAM monitor"
( while true; do (amd-smi metric --mem 2>/dev/null | grep -i used_vram || rocm-smi --showmeminfo vram 2>/dev/null | grep -i used) | head -1; sleep 3; done ) > "$ROOT/results/vram.log" 2>&1 &
MON=$!

echo "== end-to-end run (in-process)"
python3 "$ROOT/tests/run_local.py" --corpus "$KIT/mc3-corpus" --questions "$KIT/sample-questions.json" --index-dir "$ROOT/.index" 2>&1 | tee "$ROOT/results/run_local.txt"

echo "== contract run (server + thin client, as the harness does it)"
export MC3_OUTPUT_DIR="$ROOT/results/output" MC3_SOCKET=/tmp/mc3-nb.sock MC3_PID=/tmp/mc3-nb.pid
rm -rf "$MC3_OUTPUT_DIR" "$ROOT/.index"; mkdir -p "$MC3_OUTPUT_DIR"
( time python3 "$ROOT/app/app.py" --index "$KIT/mc3-corpus" ) 2>&1 | tee "$ROOT/results/index.txt"
python3 - <<PY | tee "$ROOT/results/contract.txt"
import json, subprocess, time, os
kit = "$KIT"; out = os.environ["MC3_OUTPUT_DIR"]
qs = json.load(open(os.path.join(kit, "sample-questions.json")))["queries"]
score = 0
for q in qs:
    qid = f"query_{q['n']:02d}"
    t0 = time.time()
    subprocess.run(["python3", "$ROOT/app/app.py", "--corpus", os.path.join(kit, "mc3-corpus"), "--query-id", qid, "--query", q["query"]], check=False, timeout=40)
    dt = time.time() - t0
    r = json.load(open(os.path.join(out, qid + "_output.json")))
    norm = lambda s: "".join(ch for ch in s.upper() if ch not in " -._·\t\n")
    ok = norm(r["answer"]) in {norm(q["expected_answer"])} | {norm(a) for a in q.get("answer_aliases", [])} and sorted(r["citations"]) == sorted(q["expected_citations"])
    score += 20 if ok else 0
    print(f"{q['n']:>2} {str(ok):>5} {dt:5.1f}s  {r['answer']!r} {r['citations']}")
print("CONTRACT SCORE", score, "/ 200")
PY
kill $MON 2>/dev/null || true
echo "== peak VRAM"; sort -k1 "$ROOT/results/vram.log" | tail -3 || true
pkill -f "$ROOT/app/server.py" || true
echo "done -> $ROOT/results/"
