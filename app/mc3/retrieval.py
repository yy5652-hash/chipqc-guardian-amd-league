"""Hybrid retrieval: identifier-aware BM25 + dense embeddings, fused with RRF."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config
from .chunk import Chunk

log = logging.getLogger("mc3.retrieval")

TOKEN_RE = re.compile(r"[A-Za-z0-9一-鿿][A-Za-z0-9#\-._/一-鿿]*")
IDENT_RE = re.compile(
    r"\b(?:[A-Z]{1,6}-?\d{2,6}(?:-[A-Z0-9]{1,6})*|[A-Z]{2,}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*|v?\d+\.\d+(?:\.\d+)+|"
    r"[A-Z]\d{3,6}|[A-Z]{1,3}\d{1,3}[A-Z]?|REV[-_ ]?[A-Z0-9]+|Q[1-4]\s?FY\d{2,4}|FY\d{2,4}|0x[0-9A-Fa-f]+)\b"
)
STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "are", "was", "were", "be", "by", "with", "and",
    "or", "what", "which", "who", "when", "where", "how", "does", "do", "did", "it", "its", "this", "that",
    "from", "as", "has", "have", "had", "into", "than", "then", "there", "their", "they", "them", "we", "you",
    "will", "can", "could", "should", "would", "may", "might", "per", "any", "all", "if", "not", "no", "yes",
    "many", "much", "much", "value", "number", "please", "tell", "me", "about", "according",
}


def tokenize(s: str) -> list[str]:
    toks: list[str] = []
    for m in TOKEN_RE.finditer(s.lower()):
        t = m.group(0).strip("-._/#")
        if not t:
            continue
        toks.append(t)
        parts = [p for p in re.split(r"[\-._/#]+", t) if p]
        if len(parts) > 1:
            toks.extend(parts)
        for p in parts:
            sub = re.findall(r"[a-z]+|\d+|[一-鿿]", p)
            if len(sub) > 1:
                toks.extend(sub)
    return toks


def query_tokens(q: str) -> list[str]:
    return [t for t in tokenize(q) if t not in STOP and len(t) > 1 or t.isdigit()]


def identifiers(text: str) -> list[str]:
    out: list[str] = []
    for m in IDENT_RE.finditer(text):
        v = m.group(0)
        if len(v) < 3 or v.lower() in {"the", "and"}:
            continue
        if re.fullmatch(r"\d{4}", v):  # bare years are not identifiers
            continue
        out.append(v)
    seen = set()
    uniq = []
    for v in out:
        k = v.upper()
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 1.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5)) for t, n_t in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings[t].append(i)

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return s
        qc = Counter(q)
        for t, qf in qc.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            w = idf * (1.0 + 0.5 * math.log(qf))
            for i in self.postings[t]:
                f = self.tf[i][t]
                dl = self.doc_len[i]
                s[i] += w * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


class Embedder:
    """Qwen3-Embedding via transformers: last-token pooling, left padding."""

    def __init__(self, path: Path, device: str, dtype: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(str(path), padding_side="left")
        dt = getattr(torch, dtype) if device != "cpu" else getattr(torch, __import__("os").environ.get("MC3_CPU_DTYPE", "float32"))
        self.model = AutoModel.from_pretrained(str(path), dtype=dt)
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 16, max_length: int = 1024) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        outs = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"]
                # left padding -> the last position is the last real token
                if bool((mask[:, -1] == 1).all()):
                    e = h[:, -1]
                else:
                    idx = mask.sum(dim=1) - 1
                    e = h[torch.arange(h.shape[0], device=h.device), idx]
                e = F.normalize(e.float(), dim=-1)
            outs.append(e.cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1), dtype=np.float32)

    @staticmethod
    def query_text(q: str) -> str:
        return ("Instruct: Given a question about internal technical documents (datasheets, roadmaps, part lists, "
                "bug tickets, logs, source code, diagrams), retrieve the passage that contains the answer\nQuery: " + q)


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25([tokenize(c.header() + "\n" + c.text) for c in chunks])
        self.emb: np.ndarray | None = None
        self.by_id = {c.id: c for c in chunks}

    # ----- persistence
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False))
        if self.emb is not None:
            np.save(d / "emb.npy", self.emb)

    @classmethod
    def load(cls, d: Path, embedder: Embedder | None) -> "Index":
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        idx = cls(chunks, embedder)
        p = d / "emb.npy"
        if p.exists():
            idx.emb = np.load(p)
        return idx

    def embed_all(self) -> None:
        if self.embedder is None or not self.chunks:
            return
        texts = [c.header() + "\n" + c.text for c in self.chunks]
        self.emb = self.embedder.encode(texts)

    # ----- search
    def search(self, query: str, k: int, extra_terms: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        qt = query_tokens(query)
        if extra_terms:
            for t in extra_terms:
                qt.extend(tokenize(t) * 2)  # weight identifiers
        bm = self.bm25.scores(qt)
        order_bm = np.argsort(-bm)
        rank_bm = {int(i): r for r, i in enumerate(order_bm[: max(k * 6, 60)]) if bm[i] > 0}

        rank_dn: dict[int, int] = {}
        dense_scores = None
        if self.emb is not None and self.embedder is not None:
            qtext = Embedder.query_text(query if not extra_terms else query + " " + " ".join(extra_terms))
            qv = self.embedder.encode([qtext])[0]
            dense_scores = self.emb @ qv
            order_dn = np.argsort(-dense_scores)
            rank_dn = {int(i): r for r, i in enumerate(order_dn[: max(k * 6, 60)])}

        fused: dict[int, float] = defaultdict(float)
        K = 60.0
        for i, r in rank_bm.items():
            fused[i] += 1.0 / (K + r)
        for i, r in rank_dn.items():
            fused[i] += 1.0 / (K + r)
        # exact identifier match bonus: a chunk that literally contains an identifier from the query
        idents = [x.lower() for x in identifiers(query)] + [x.lower() for x in (extra_terms or [])]
        if idents:
            for i in list(fused.keys()):
                low = self.chunks[i].text.lower()
                hits = sum(1 for x in idents if x and x in low)
                if hits:
                    fused[i] += 0.01 * hits
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return [(self.chunks[i], s) for i, s in ranked[:k]]
