"""Answering pipeline: retrieve -> (hop 2 on identifiers) -> value + evidence -> citation necessity -> verification."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from . import config
from .chunk import Chunk, build_chunks
from .ingest import ingest_corpus
from .retrieval import Index, Embedder, identifiers
from .util import Deadline, clean_answer, extract_json, normalize_answer, text_contains_value

log = logging.getLogger("mc3.pipeline")

BAD_STATUS = {"WITHDRAWN", "SUPERSEDED", "OBSOLETE", "DEPRECATED", "RETIRED", "ARCHIVED", "REVOKED",
              "CANCELLED", "CANCELED", "RESCINDED", "VOID", "OLD"}

# identifiers that can link two documents: tickets, error codes, part/serial numbers, pins.
# Version numbers, quarters and bare numbers are deliberately NOT chain identifiers.
STRONG_IDENT_RE = re.compile(
    r"\b(?:[A-Z]{2,6}-\d{2,6}(?:-[A-Z0-9]{1,6})*|[A-Z]{2,}-[A-Z]{2,}-\d{2,6}(?:-[A-Z0-9]{1,6})*|[A-Z]{1,3}\d{3,6}[A-Z]?|"
    r"[A-Z]\d{2}|#\d{3,6}|(?:SN|S/N|SERIAL)[:\s-]*[A-Z0-9-]{5,})\b"
)

SYSTEM = """You answer questions about a private collection of internal technical documents: datasheets, roadmaps, part lists, bug and ticket databases, production logs, release notes, source code, diagrams and photos. The products in these documents are fictional. Nothing you remember from training applies; only the excerpts provided count as evidence.

Rules:
1. Answer with the VALUE ONLY: a number, a part number, a version, a quarter, a code, a pin name, a revision, a name. No sentence, no explanation, no restatement of the question, no units unless the unit is part of an identifier. Keep qualifiers that identify the value: "Q3 FY27" not "Q3", "REV-C2" not "C2", "4.3.2" not "4.3", "ORR-FAN-2214-B" not "2214".
2. The value must be stated for EXACTLY the item, product, part, revision, quantity tier and condition the question asks about. A value that belongs to a different part, product, row, column, tier or situation is NOT the answer.
3. If the excerpts do not state the answer, set "answer" to "" (empty string). Never guess, never estimate, never answer from memory, never answer from an excerpt that only discusses the topic.
4. Excerpts whose DOCUMENT STATUS is WITHDRAWN, SUPERSEDED, OBSOLETE, DEPRECATED, RETIRED or ARCHIVED are not valid sources unless the question explicitly asks about that old document. Prefer the current revision. If an excerpt says a value was superseded or replaced, use the replacement.
5. "evidence" lists ONLY the excerpt ids that are strictly necessary: the excerpt that states the value, plus — only when the question itself does not give it — the excerpt that supplied the identifier (ticket id, error code, part number, serial) you needed to find that value. Do NOT list excerpts that merely confirm, discuss, summarise or point to the answer. Usually one id; sometimes two.
6. If the excerpts name an identifier (ticket id, error code, part number, pin, serial, SKU) whose associated value you need but which is NOT in the excerpts, put the identifier(s) in "lookup" and leave "answer" empty. "lookup" holds identifiers from the documents, never excerpt ids.
7. When an image is attached, read the value directly from the image; the excerpt id it belongs to is stated next to it.

Respond with a single JSON object and nothing else:
{"answer": "<value or empty>", "evidence": ["<excerpt id>", ...], "lookup": ["<identifier>", ...]}"""

VERIFY = """You are auditing one proposed answer against the excerpt(s) it was taken from. Reply with JSON only:
{"supported": true} or {"supported": false}

"supported" is true only if an excerpt states that the proposed value is the quantity the question asks for, for EXACTLY the item named in the question: the same product or part number, the same revision, the same quantity tier or condition, and the same meaning of the column or field. Units, degree signs and formatting differences do not matter.
The excerpts may form a chain: one excerpt links the question to an identifier (a ticket id, error code, part number, serial) and another excerpt gives the value for that identifier. A complete chain counts as supported.
It is false if the value belongs to a different part, product, row, column, tier or situation, if the excerpt only mentions the number in another context, or if the excerpts do not actually contain the value for that item."""


def _contains(norm_answer: str, text: str, raw_answer: str | None = None) -> bool:
    """Does the excerpt carry the value? Token-boundary match on the raw answer when available."""
    if not norm_answer:
        return False
    if raw_answer:
        return text_contains_value(raw_answer, text)
    return norm_answer in normalize_answer(text)


def strong_identifiers(text: str) -> set[str]:
    out = set()
    for m in STRONG_IDENT_RE.finditer(text):
        v = m.group(0).upper().strip()
        if len(v) >= 3 and not re.fullmatch(r"\d+", v):
            out.add(v)
    return out

_GENERIC_PATH_WORDS = {"support", "docs", "doc", "data", "files", "file", "misc", "txt", "pdf", "csv", "xlsx", "xlsm", "docx",
                       "png", "jpg", "jpeg", "tif", "tiff", "gif", "bmp", "webp", "py", "md", "the", "and", "for", "new", "old",
                       "final", "copy", "internal", "public", "export", "exports", "report", "reports"}
_KIND_WORDS = {
    "image": {"image", "photo", "photograph", "picture", "diagram", "label", "drawing", "figure", "scan", "screenshot"},
    "row": {"spreadsheet", "sheet", "table", "database", "db", "list", "csv", "xlsx", "export", "register", "inventory"},
    "table": {"table", "roadmap", "document"},
    "code": {"code", "source", "script", "module", "constant", "function"},
}


def question_mentions_file(question: str, file: str, kinds: set[str]) -> bool:
    """Does the question point at this document (by a word of its path or by its kind)?"""
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    q |= {w[:-1] for w in q if w.endswith("s") and len(w) > 3}
    toks = [t for t in re.split(r"[^a-z0-9]+", file.lower()) if len(t) >= 3 and t not in _GENERIC_PATH_WORDS and not t.isdigit()]
    toks += [t[:-1] for t in toks if t.endswith("s")]
    if any(t in q for t in toks):
        return True
    if "log" in q and (file.lower().endswith(".log") or "log" in file.lower()):
        return True
    for k in kinds:
        if _KIND_WORDS.get(k, set()) & q:
            return True
    return False


class Engine:
    def __init__(self):
        self.vlm = None
        self.embedder = None
        self.index: Index | None = None
        self.corpus: Path | None = None
        self._ident_df: dict[str, int] = {}
        self._load_models()

    # ------------------------------------------------------------------ models
    def _load_models(self) -> None:
        if config.STUB:
            import os
            if os.environ.get("MC3_STUB") == "oracle":
                from .oracle import OracleVLM
                self.vlm = OracleVLM()
            else:
                from .llm import StubVLM
                self.vlm = StubVLM()
            self.embedder = None
            log.warning("STUB mode: no models loaded")
            return
        from .llm import VLM
        t0 = time.monotonic()
        self.embedder = Embedder(config.EMB_PATH, config.DEVICE, config.DTYPE)
        log.info("embedder loaded in %.1fs", time.monotonic() - t0)
        self.vlm = VLM(config.VLM_PATH, config.DEVICE, config.DTYPE)
        if config.DEVICE != "cpu":
            try:
                import torch
                torch.cuda.synchronize()
                log.info("GPU memory allocated after load: %.2f GiB", torch.cuda.memory_allocated() / 2**30)
            except Exception as e:
                log.warning("cuda check: %s", e)

    # ------------------------------------------------------------------ index
    def _compute_ident_df(self) -> None:
        """How many distinct files mention each strong identifier (product names are everywhere)."""
        df: dict[str, set] = {}
        for c in self.index.chunks if self.index else []:
            for ident in strong_identifiers(c.text):
                df.setdefault(ident, set()).add(c.file)
        self._ident_df = {k: len(v) for k, v in df.items()}

    def build_index(self, corpus: Path, budget_s: float | None = None) -> dict:
        budget = Deadline(budget_s or config.INDEX_BUDGET_S)
        corpus = Path(corpus)
        self.corpus = corpus
        docs = ingest_corpus(corpus)
        n_img = 0
        for doc in docs:
            if doc.kind == "image" and doc.segments:
                if not budget.ok(45):
                    log.warning("index budget nearly spent; skipping image %s", doc.rel)
                    continue
                try:
                    seg = doc.segments[0]
                    seg.text = self.vlm.transcribe(seg.image_path) if seg.image_path else ""
                    n_img += 1
                    log.info("transcribed %s: %d chars", doc.rel, len(seg.text))
                except Exception as e:  # noqa: BLE001
                    log.warning("transcribe failed %s: %s", doc.rel, e)
            if doc.kind == "pdf" and doc.ocr_pages:
                try:
                    import fitz
                    from .ingest import Segment
                    render_dir = config.INDEX_DIR / "render"
                    render_dir.mkdir(parents=True, exist_ok=True)
                    d = fitz.open(doc.abs)
                    for pno in doc.ocr_pages:
                        if not budget.ok(45):
                            break
                        try:
                            pix = d[pno].get_pixmap(dpi=140)
                            out = render_dir / (re.sub(r"[^A-Za-z0-9]+", "_", doc.rel) + f"_p{pno + 1}.png")
                            pix.save(str(out))
                            text = self.vlm.transcribe(str(out))
                            doc.segments.append(Segment(text=text, loc=f"page {pno + 1} (image OCR)", kind="image_page",
                                                        image_path=str(out), page=pno + 1))
                            n_img += 1
                        except Exception as e:  # noqa: BLE001
                            log.warning("pdf OCR failed %s p%d: %s", doc.rel, pno + 1, e)
                    d.close()
                except Exception as e:  # noqa: BLE001
                    log.warning("pdf OCR setup failed %s: %s", doc.rel, e)
        chunks = build_chunks(docs)
        self.index = Index(chunks, self.embedder)
        t0 = time.monotonic()
        try:
            self.index.embed_all()
        except Exception as e:  # noqa: BLE001 - BM25 alone still works
            log.error("embedding failed, lexical only: %s", e)
            self.index.emb = None
        log.info("embedded %d chunks in %.1fs", len(chunks), time.monotonic() - t0)
        self._compute_ident_df()
        try:
            self.index.save(config.INDEX_DIR)
            (config.INDEX_DIR / "corpus_path.txt").write_text(str(corpus))
        except Exception as e:  # noqa: BLE001
            log.warning("index save failed: %s", e)
        errs = {d.rel: d.error for d in docs if d.error}
        return {"files": len(docs), "chunks": len(chunks), "images": n_img, "skipped": errs,
                "seconds": round(budget.elapsed(), 1)}

    def try_load_index(self) -> bool:
        try:
            if (config.INDEX_DIR / "chunks.json").exists():
                self.index = Index.load(config.INDEX_DIR, self.embedder)
                p = config.INDEX_DIR / "corpus_path.txt"
                if p.exists():
                    self.corpus = Path(p.read_text().strip())
                self._compute_ident_df()
                log.info("index loaded from disk: %d chunks", len(self.index.chunks))
                return True
        except Exception as e:  # noqa: BLE001
            log.warning("index load failed: %s", e)
        return False

    # ------------------------------------------------------------------ generation
    def _parts(self, ctx: list[Chunk], attach_images: bool, max_images: int = 3) -> list[dict]:
        parts: list[dict] = []
        lines = ["Excerpts from the document collection (each starts with its id and file path):", ""]
        for c in ctx:
            lines.append(f"=== [{c.id}] {c.header()} ===")
            lines.append(c.text if c.text.strip() else "(no legible text)")
            lines.append("")
        parts.append({"type": "text", "text": "\n".join(lines)})
        if attach_images and not config.STUB:
            from .llm import load_image
            n = 0
            for c in ctx[:8]:
                if n >= max_images:
                    break
                if c.kind in {"image", "image_page"} and c.image_path:
                    try:
                        img = load_image(c.image_path)
                    except Exception as e:  # noqa: BLE001
                        log.warning("image load failed %s: %s", c.image_path, e)
                        continue
                    parts.append({"type": "text", "text": f"Image attached for excerpt [{c.id}] (FILE: {c.file}):"})
                    parts.append({"type": "image", "image": img})
                    n += 1
        return parts

    def _ask(self, question: str, ctx: list[Chunk]) -> dict:
        parts = self._parts(ctx, attach_images=True)
        parts.append({"type": "text", "text": f"\nQuestion: {question}\n\nJSON:"})
        t0 = time.monotonic()
        raw = self.vlm.chat(SYSTEM, parts, max_new_tokens=120)
        obj = extract_json(raw)
        log.info("llm %.1fs ctx=%d -> %s", time.monotonic() - t0, len(ctx), raw[:300].replace("\n", " "))
        ev = obj.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]
        lk = obj.get("lookup") or []
        if isinstance(lk, str):
            lk = [lk]
        q_low = question.lower()
        lookup = []
        for x in lk:
            s = str(x).strip()
            if not s or re.fullmatch(r"c\d+", s) or s.lower() in q_low:
                continue
            lookup.append(s)
        return {"answer": clean_answer(obj.get("answer", "")), "evidence": [str(e).strip("[] ") for e in ev],
                "lookup": lookup, "raw": raw}

    def _verify(self, question: str, answer: str, chunks: list[Chunk]) -> bool | None:
        """True/False from the model, None if it could not be parsed (treated as supported)."""
        parts = self._parts(chunks, attach_images=True, max_images=2)
        parts.append({"type": "text", "text": f"\nQuestion: {question}\nProposed answer: {answer}\n\nJSON:"})
        t0 = time.monotonic()
        raw = self.vlm.chat(VERIFY, parts, max_new_tokens=24)
        obj = extract_json(raw)
        log.info("verify %.1fs -> %s", time.monotonic() - t0, raw[:120].replace("\n", " "))
        v = obj.get("supported")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes"}
        return None

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, budget_s: float | None = None) -> dict:
        dl = Deadline(budget_s or config.SERVER_SOFT_BUDGET_S)
        if self.index is None or not self.index.chunks:
            return {"answer": "", "citations": [], "confidence": 0.0, "note": "no index"}
        idx = self.index
        ctx = [c for c, _ in idx.search(question, config.TOP_K)]
        ctx_ids = {c.id for c in ctx}
        r = self._ask(question, ctx)
        answer, evidence, lookup = r["answer"], [e for e in r["evidence"] if e in ctx_ids], r["lookup"]
        hop_used = False

        # ---- hop 2: identifiers the model asked for, or found in the top excerpts
        if (lookup or not answer) and dl.ok(5):
            terms = [t for t in lookup if len(t) >= 2][:4]
            q_idents = {x.upper() for x in identifiers(question)}
            if not terms:
                cand: list[str] = []
                for c in ctx[:4]:
                    for x in identifiers(c.text):
                        if x.upper() not in q_idents and x not in cand:
                            cand.append(x)
                terms = cand[:6]
            if terms:
                extra = [c for c, _ in idx.search(question, config.HOP2_K, extra_terms=terms)]
                new = [c for c in extra if c.id not in ctx_ids]
                if new:
                    merged = (ctx + new)[: config.MAX_CONTEXT_CHUNKS]
                    r2 = self._ask(question, merged)
                    if r2["answer"]:
                        ctx = merged
                        ctx_ids = {c.id for c in ctx}
                        answer = r2["answer"]
                        evidence = [e for e in r2["evidence"] if e in ctx_ids]
                        hop_used = True
                        log.info("hop2 with %s -> %r", terms, answer)
        if not answer:
            return {"answer": "", "citations": [], "confidence": 0.0}

        # ---- citations: necessity, not relevance
        by_id = {c.id: c for c in ctx}
        rank = {c.file: i for i, c in reversed(list(enumerate(ctx)))}  # best (lowest) rank per file
        ev_chunks = [by_id[e] for e in evidence if e in by_id]
        cited: list[str] = []
        for c in ev_chunks:
            if c.file not in cited:
                cited.append(c.file)
        norm = normalize_answer(answer)
        file_chunks: dict[str, list[Chunk]] = {}
        for c in ctx:
            file_chunks.setdefault(c.file, []).append(c)

        def is_image(c: Chunk) -> bool:
            return c.kind in {"image", "image_page"}

        def value_chunks(f: str) -> list[Chunk]:
            """chunks of file f that carry the value (image chunks count when cited or illegible)."""
            out = []
            for c in file_chunks.get(f, []):
                if _contains(norm, c.text, answer) or (is_image(c) and (any(e.id == c.id for e in ev_chunks) or not c.text.strip())):
                    out.append(c)
            return out

        def status_bad(f: str) -> bool:
            cs = file_chunks.get(f, [])
            return bool(cs) and cs[0].status.upper() in BAD_STATUS

        if not cited:
            # the model gave an answer but no usable ids: take the best-ranked excerpt that carries the value
            for c in ctx:
                if (_contains(norm, c.text, answer) or is_image(c)) and not status_bad(c.file):
                    cited = [c.file]
                    break
            if not cited:
                log.info("answer %r not found in any excerpt: refusing", answer)
                return {"answer": "", "citations": [], "confidence": 0.0}

        A = [f for f in cited if value_chunks(f)]                 # files that carry the value
        if not A:
            alt = [c.file for c in ctx if _contains(norm, c.text, answer) and not status_bad(c.file)]
            if alt:
                A = [alt[0]]
            elif any(is_image(c) for f in cited for c in file_chunks.get(f, [])):
                A = [f for f in cited if any(is_image(c) for c in file_chunks.get(f, []))]
            else:
                log.info("cited files do not contain %r and no image: refusing", answer)
                return {"answer": "", "citations": [], "confidence": 0.0}
        A_good = [f for f in A if not status_bad(f)]
        if A_good:
            A = A_good

        # ---- chain links: the file that supplied the identifier used to reach the value.
        # Directional: a chain link is reached from the question (ranks better than the value file),
        # shares a strong, document-specific identifier with the value chunk, and that identifier is
        # not given in the question itself. Applies to cited and uncited files alike.
        q_strong = strong_identifiers(question)
        chain: list[str] = []
        chain_chunks: list[Chunk] = []
        for f in A:
            a_idents: set[str] = set()
            for c in value_chunks(f):
                a_idents |= strong_identifiers(c.text)
            a_idents -= q_strong
            a_idents.discard(norm)
            a_idents = {i for i in a_idents if self._ident_df.get(i, 1) <= 2 and normalize_answer(i) != norm}
            if not a_idents:
                continue
            for g, cs in file_chunks.items():
                if g == f or g in A or g in chain or status_bad(g):
                    continue
                shared = set()
                link_chunks = []
                for c in cs:
                    got = strong_identifiers(c.text) & a_idents
                    if got:
                        shared |= got
                        link_chunks.append(c)
                if not shared:
                    continue
                # directional evidence that the question was answered THROUGH this file:
                reached_first = rank.get(g, 999) < rank.get(f, 999)          # retrieved ahead of the value file
                pointed_at = question_mentions_file(question, g, {c.kind for c in cs})  # "the production log shows..."
                model_needed = (g in cited) and hop_used                      # cited, and a lookup hop was required
                if reached_first or pointed_at or model_needed:
                    log.info("chain link %s -> %s via %s (rank=%s mention=%s hop=%s)", g, f, sorted(shared),
                             reached_first, pointed_at, model_needed)
                    chain.append(g)
                    chain_chunks.extend(link_chunks[:1])
                else:
                    log.info("shared identifier %s with %s but no directional evidence: not a chain link", sorted(shared), g)
        dropped = [f for f in cited if f not in A and f not in chain]
        if dropped:
            log.info("dropping over-citations %s", dropped)

        # ---- verification: the value must belong to the item asked about (anti-guessing)
        if dl.ok(4):
            vchunks = list(chain_chunks[:2])
            for f in A[:2]:
                vchunks.extend(value_chunks(f)[:1] or file_chunks[f][:1])
            try:
                ok = self._verify(question, answer, vchunks)
            except Exception as e:  # noqa: BLE001
                log.warning("verify failed: %s", e)
                ok = None
            if ok is False:
                log.info("verification rejected %r from %s: refusing", answer, A)
                return {"answer": "", "citations": [], "confidence": 0.0}

        # ---- redundancy among value-carrying files: ablate, keep those whose removal changes the answer
        if len(A) >= 2 and dl.ok(4):
            necessary: list[str] = []
            for f in A[:3]:
                if not dl.ok(3):
                    necessary.append(f)
                    continue
                ctx_minus = [c for c in ctx if c.file != f]
                ra = self._ask(question, ctx_minus)
                same = normalize_answer(ra["answer"]) == norm
                log.info("ablation without %s -> %r (same=%s)", f, ra["answer"], same)
                if not same:
                    necessary.append(f)
            A = (necessary + A[3:]) if necessary else [A[0]]
        elif len(A) >= 2:
            A = [A[0]]

        citations: list[str] = []
        for f in chain + A:
            if f not in citations:
                citations.append(f)
        # keep a stable, deterministic order: by retrieval rank
        citations.sort(key=lambda f: rank.get(f, 999))
        conf = 0.9 if not hop_used else 0.8
        if dropped or (chain and not all(g in cited for g in chain)):
            conf -= 0.1
        return {"answer": answer, "citations": citations, "confidence": round(max(0.05, conf), 2)}
