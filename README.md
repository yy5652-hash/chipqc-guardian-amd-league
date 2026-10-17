# ChipQC Guardian — citation-exact RAG and vision-language OCR on AMD ROCm (AMD AI League)

Two containers that run on AMD Instinct GPUs with ROCm, built on the mandated `rocm/pytorch` image with every weight inside:

* **Match 3, RAG** (`app/`, `Dockerfile`, image `ghcr.io/yy5652-hash/chipqc-mc3:v1`): indexes a folder of mixed enterprise
  documents once, then answers each question with the bare value and the exact set of files it came from, returning an
  empty answer when the corpus does not contain it. 200/200 on the official sample corpus, 14/14 on an adversarial corpus.
* **Match 2, OCR** (`app_mc2/`, `Dockerfile.mc2`, image `ghcr.io/yy5652-hash/chipqc-mc3:ocr-v1`): reads US and Chinese
  licence plates, stop, speed-limit, advisory and work-zone signs. 10/10 on the official samples, 20/20 on noisy, blurred,
  dark, glary, rotated, small-in-scene and low-resolution variants. It reuses every model layer of the RAG image.

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
| `app_mc2/app.py`, `app_mc2/server.py` | Match 2 thin client (`--input-image`) and resident OCR server |
| `app_mc2/ocr.py` | typed transcription JSON, grounding crop, enhanced re-read, vote, convention-aware cleaners |
| `tests/run_mc2.py` | Match 2 run over a sample list, scored with the grader's normalisation |

## Models

* `Qwen/Qwen3-VL-8B-Instruct` (bf16) — generation, image reading, verification
* `Qwen/Qwen3-Embedding-0.6B` — dense retrieval

Both are copied into `/models` at build time (`models/` is not committed; download with `huggingface_hub`).

## Match 2 contract

`python3 /app/app.py --input-image /app/input/image_01.png` writes `/app/output/image_01_output.json` =
`{"text": "7ABC123", "confidence": 0.94}`. Answers follow the official convention: plate characters only (no state
name), Chinese plates keep the province character, `SPEED LIMIT 65`, digits-only advisory plaques, multi-line signs
joined top to bottom.

## Build

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false -t chipqc-mc3:v1 .
docker build --platform linux/amd64 --provenance=false --sbom=false -f Dockerfile.mc2 -t chipqc-mc3:ocr-v1 .
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
