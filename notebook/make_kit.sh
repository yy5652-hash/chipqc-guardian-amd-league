#!/bin/bash
# Bundle app + tests + the official sample kit into one tarball for upload to the AMD notebook.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
KIT="${1:?path to the unpacked mc3-starter-kit directory}"
STAGE="$(mktemp -d)/mc3-kit"; mkdir -p "$STAGE/sample"
cp -R "$HERE/app" "$HERE/tests" "$HERE/notebook" "$STAGE/"
LOCKED="$KIT/mc3-corpus/vendor/internal_audit.txt"
chmod u+r "$LOCKED" 2>/dev/null || true
cp -R "$KIT/mc3-corpus" "$KIT/sample-questions.json" "$STAGE/sample/"
chmod 000 "$LOCKED" 2>/dev/null || true
chmod 644 "$STAGE/sample/mc3-corpus/vendor/internal_audit.txt" 2>/dev/null || true
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
rm -f "$STAGE/notebook/mc3-kit.tgz"
tar -C "$(dirname "$STAGE")" -czf "$HERE/notebook/mc3-kit.tgz" mc3-kit
ls -la "$HERE/notebook/mc3-kit.tgz"
