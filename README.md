# ChipQC Guardian — citation-exact RAG for the AMD AI League (Match 3)

A containerised retrieval-augmented question-answering system that runs on AMD Instinct GPUs with ROCm.
Given a folder of mixed enterprise documents it indexes everything once, then answers each question with the
bare value and the exact set of files the value came from, refusing when the corpus does not contain the answer.

## Layout

| Path | Role |
|---|---|
| `Dockerfile` | final stage on `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`; weights shipped in the image |
| `app/app.py` | thin client for the Match contract (`--index`, then `--corpus/--query-id/--query`) |
| `app/server.py` | resident model + index server behind a unix socket (started by the container CMD) |
| `app/mc3/ingest.py` | defensive corpus walk; pdf / docx / xlsx / csv / txt / log / py / image parsers |
| `app/mc3/retrieval.py` | identifier-aware BM25 + Qwen3-Embedding dense retrieval, RRF fusion |
| `app/mc3/pipeline.py` | value-only answering, identifier hop, citation necessity, verification |
| `app/mc3/llm.py` | Qwen3-VL wrapper (text + images) |
| `tests/run_local.py` | in-process run over a corpus + question file, scored like the grader |
| `tests/synthetic_corpus.py` | builds a deliberately nasty corpus (encrypted, unreadable, image-only pages, odd encodings) |
| `notebook/` | kit to validate on an AMD hackathon JupyterLab GPU session |

## Models

* `Qwen/Qwen3-VL-8B-Instruct` (bf16) — generation, image reading, verification
* `Qwen/Qwen3-Embedding-0.6B` — dense retrieval

Both are copied into `/models` at build time (`models/` is not committed; download with `huggingface_hub`).

## Build

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false -t chipqc-mc3:v1 .
```

`pip install --no-deps` with the fully resolved closure in `app/requirements.txt` keeps the ROCm torch wheel intact.

## Local checks

```bash
# logic only, with an answer-key oracle in place of the model
MC3_STUB=oracle MC3_ORACLE_FILE=sample-questions.json python3 tests/run_local.py --corpus mc3-corpus --questions sample-questions.json --stub
# real models on CPU (slow) or GPU
python3 tests/run_local.py --corpus mc3-corpus --questions sample-questions.json [--device cpu]
```

## Design notes

* **Two-phase contract.** `--index` parses, transcribes images, embeds and persists; each `--query` is a new process
  that talks to the resident server and always writes its JSON before the per-question budget.
* **Citations are a necessity test.** Files that merely discuss the topic are dropped; a second file is kept only as a
  chain link (it supplied a ticket / part / error identifier that led to the value, with directional evidence) or when
  ablation shows the answer changes without it. Withdrawn and superseded documents are never sources.
* **No guessing.** A verification pass checks that the value belongs to exactly the item asked; otherwise the answer is
  empty with no citations.
