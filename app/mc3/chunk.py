from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from . import config
from .ingest import FileDoc


@dataclass
class Chunk:
    id: str
    file: str
    text: str
    loc: str
    kind: str
    status: str
    image_path: str | None = None

    def header(self) -> str:
        h = f"FILE: {self.file}"
        if self.status:
            h += f"  [DOCUMENT STATUS: {self.status}]"
        if self.loc:
            h += f"  ({self.loc})"
        return h

    def to_dict(self) -> dict:
        return asdict(self)


_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_text(text: str, max_chars: int, overlap: int, keep_lines: bool) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    units = text.splitlines() if keep_lines else re.split(r"(?<=[.!?])\s+|\n", text)
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for u in units:
        u = u.rstrip()
        if not u:
            continue
        if size + len(u) + 1 > max_chars and buf:
            pieces.append("\n".join(buf))
            # overlap: carry the tail units forward
            tail: list[str] = []
            tsize = 0
            for v in reversed(buf):
                if tsize + len(v) > overlap:
                    break
                tail.insert(0, v)
                tsize += len(v) + 1
            buf = tail
            size = tsize
        if len(u) > max_chars:  # a single enormous line: hard split
            for i in range(0, len(u), max_chars):
                pieces.append(u[i:i + max_chars])
            buf, size = [], 0
            continue
        buf.append(u)
        size += len(u) + 1
    if buf:
        pieces.append("\n".join(buf))
    return pieces


def build_chunks(docs: list[FileDoc]) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0
    for doc in docs:
        if doc.error and not doc.segments:
            continue
        for seg in doc.segments:
            keep_lines = seg.kind in {"row", "table", "code", "image", "image_page"}
            pieces = split_text(seg.text, config.CHUNK_CHARS, config.CHUNK_OVERLAP, keep_lines)
            if not pieces and seg.kind in {"image", "image_page"}:
                pieces = ["(no legible text)"]
            for piece in pieces:
                n += 1
                chunks.append(Chunk(id=f"c{n}", file=doc.rel, text=piece, loc=seg.loc, kind=seg.kind,
                                    status=doc.status, image_path=seg.image_path))
    return chunks
