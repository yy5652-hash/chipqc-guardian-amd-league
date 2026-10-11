"""Oracle stand-in for the VLM: answers from an answer key so the retrieval and
citation logic can be exercised without a model. Test-only (MC3_STUB=oracle)."""
from __future__ import annotations

import json
import os
import re

from .util import normalize_answer


class OracleVLM:
    def __init__(self, *a, **k):
        self.device = "cpu"
        path = os.environ.get("MC3_ORACLE_FILE")
        self.key: dict[str, dict] = {}
        if path:
            data = json.loads(open(path).read())
            for q in data["queries"]:
                self.key[q["query"].strip().lower()] = q

    def transcribe(self, image_path: str) -> str:
        # pretend the vision model read the sample images
        if "pinout" in image_path:
            return "TQ-40 backplane pinout\nA1: GND | A2: 12V | B13: PRSNT# | B14: THERM_ALERT# | B15: SMB_CLK"
        if "asset_label" in image_path:
            return "ORRERY SYSTEMS  ASSET LABEL\nMODEL: TQ-40  REV: REV-C2  S/N: OS-44812"
        return ""

    def chat(self, system, user_parts, max_new_tokens=160) -> str:
        text = "\n".join(p.get("text", "") for p in user_parts if p.get("type") == "text")
        if "Proposed answer:" in text:  # verification call
            return '{"supported": true}' 
        m = re.search(r"Question: (.*?)\n\nJSON:", text, re.S)
        q = (m.group(1) if m else "").strip().lower()
        entry = self.key.get(q)
        if not entry or not entry.get("expected_answer"):
            return json.dumps({"answer": "", "evidence": [], "lookup": []})
        ans = entry["expected_answer"]
        norm = normalize_answer(ans)
        # excerpts in context: id -> (file, body)
        exc = re.findall(r"=== \[(c\d+)\] FILE: (\S+)(.*?)===\n(.*?)(?=\n=== \[|\n\nImage attached|\nQuestion:)", text, re.S)
        files_in_ctx = {cid: f for cid, f, _, _ in exc}
        containing = [cid for cid, f, _, body in exc if norm in normalize_answer(body)]
        expected_files = set(entry.get("expected_citations", []))
        if not containing:
            # simulate a model asking for the missing identifier (multi-hop)
            idents = re.findall(r"\b[A-Z]{2,5}-\d{3,5}\b", text)
            return json.dumps({"answer": "", "evidence": [], "lookup": sorted(set(idents))[:2]})
        # a slightly over-citing model: everything that carries the value, plus expected chain files present
        ev = list(containing)
        for cid, f in files_in_ctx.items():
            if f in expected_files and cid not in ev:
                ev.append(cid)
        return json.dumps({"answer": ans, "evidence": ev, "lookup": []})
