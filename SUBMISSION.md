# lablab submission draft — Match 3

**Title (≤50):** ChipQC Guardian: Citation-Exact RAG on ROCm

**Long description (≥600 chars):**

ChipQC Guardian is a containerised retrieval-augmented question-answering system for Match 3 of the AMD AI League, built on the mandated rocm/pytorch base and shipped with all weights inside the image. Pointed at a folder of mixed enterprise documents — PDF datasheets (including withdrawn revisions), DOCX roadmaps with tables, XLSX part lists, CSV ticket databases, production logs, release notes, Python source and diagrams or photographs — it indexes everything once, then answers each question with the bare value and the exact set of files the value came from.

How it works on AMD Instinct GPUs: a resident server process holds Qwen3-VL-8B-Instruct (bf16) and Qwen3-Embedding-0.6B behind a unix socket, so the per-question invocation is a thin client that returns in seconds and never loads a model. Ingestion is defensive: an empty directory, an unreadable (mode 000) file, an encrypted PDF and an unknown binary are each skipped per file without stopping the walk. Spreadsheets and CSVs are indexed row-by-row with their headers so a ticket number lands on the row that carries its value; Python files contribute their constants and parameter defaults; images and image-only PDF pages are transcribed by the vision model at index time and re-read directly from the image at answer time. Retrieval fuses an identifier-aware BM25 with dense embeddings; a second retrieval hop follows identifiers the model asks for, which is what makes two-file chains (log → ticket → fixed-in version) answerable. Citations are then filtered for necessity rather than relevance: withdrawn or superseded documents are dropped, chain links are kept only when they share an identifier with the file carrying the value, and redundant sources are ablated so a single primary source remains. When the value is not in any excerpt the system returns an empty answer and no citations — it never guesses and never answers from what the model remembers.

**Categories:** RAG · Vision-Language · Enterprise AI
**Technologies:** AMD ROCm, PyTorch, Qwen3-VL-8B-Instruct, Qwen3-Embedding-0.6B, Hugging Face Transformers, PyMuPDF, python-docx, openpyxl, Docker

**Mini Challenge 3 Image:** `<registry>/<repo>:<tag>` (public, pull without credentials)
