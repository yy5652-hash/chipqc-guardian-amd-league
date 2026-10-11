#!/bin/bash
# Build the submission image on the mandated ROCm base, verify the hard gates we can
# check locally, and push it to a public registry.
#
#   ./build_and_push.sh ghcr.io/<user>/chipqc-mc3:v1
#
# Prerequisites: base image pulled, models/ populated, `docker login ghcr.io`
# (gh token needs write:packages: `gh auth refresh -h github.com -s write:packages,read:packages`).
set -euo pipefail
REF="${1:?image reference, e.g. ghcr.io/user/chipqc-mc3:v1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for f in models/Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors models/Qwen3-Embedding-0.6B/model.safetensors; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
echo "== build"; docker build --platform linux/amd64 -t "$REF" .
echo "== torch is still the ROCm build"
docker run --rm --platform linux/amd64 "$REF" python3 -c "import torch; v=torch.__version__; print(v); assert '+rocm' in v"
echo "== uncompressed size (limit 60 GiB)"
docker history --no-trunc --format '{{.Size}}' "$REF" | python3 -c "
import sys
tot=0
for l in sys.stdin:
    l=l.strip()
    if not l or l=='0B': continue
    n=float(''.join(c for c in l if c.isdigit() or c=='.')); u=''.join(c for c in l if c.isalpha())
    mult={'B':1,'kB':1e3,'MB':1e6,'GB':1e9,'KB':1e3,'MiB':2**20,'GiB':2**30}[u]
    tot+=n*mult
print(f'{tot/2**30:.2f} GiB uncompressed'); sys.exit(0 if tot/2**30 < 60 else 1)"
echo "== base layer identity"
docker inspect --format '{{json .RootFS.Layers}}' rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0 > /tmp/base-layers.json
docker inspect --format '{{json .RootFS.Layers}}' "$REF" > /tmp/img-layers.json
python3 -c "
import json; b=json.load(open('/tmp/base-layers.json')); i=json.load(open('/tmp/img-layers.json'))
assert i[:len(b)]==b, 'lower layers differ from the base image!'; print('base layers intact:', len(b), 'of', len(i))"
echo "== contract smoke test with no network (stub model: no GPU on this host)"
NAME=mc3smoke-$$
docker run -d --network none --name "$NAME" -e MC3_STUB=1 "$REF" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker exec "$NAME" mkdir -p /app/corpus /app/output
docker cp "${KIT:-../external/mc3-starter-kit}/mc3-corpus/." "$NAME:/app/corpus"
time docker exec "$NAME" python3 /app/app.py --index /app/corpus
time docker exec "$NAME" python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "What is the maximum junction temperature of the TQ-40?"
docker exec "$NAME" cat /app/output/query_01_output.json; echo
docker exec "$NAME" python3 -c "import json; d=json.load(open('/app/output/query_01_output.json')); assert isinstance(d['answer'],str) and isinstance(d['citations'],list); print('shape ok')"
echo "== push"; docker push "$REF"
echo "pushed $REF  (set the package to PUBLIC in the registry UI before submitting)"
