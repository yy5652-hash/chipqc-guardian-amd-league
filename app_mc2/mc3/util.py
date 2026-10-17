from __future__ import annotations

import json
import logging
import re
import sys
import time

from . import config


def setup_logging(name: str = "mc3") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        log.addHandler(h)
        try:
            fh = logging.FileHandler(config.LOG_PATH)
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except Exception:
            pass
        log.setLevel(logging.INFO)
    return log


_NORM_DROP = re.compile(r"[\s\-\.·_]+")


def normalize_answer(s: str) -> str:
    """The grader's comparison: uppercase, strip whitespace and - . · _ ."""
    return _NORM_DROP.sub("", (s or "")).upper()


_UNIT_WORDS = (
    r"°c|°f|degc|degrees?(?:\s*c(?:elsius)?)?|c|f|k|ms|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?|days?|weeks?|"
    r"units?|pcs|pieces?|usd|eur|gbp|cny|rmb|dollars?|cents?|%|percent|"
    r"w|kw|mw|v|mv|kv|a|ma|ua|ah|mah|ohm|ohms|hz|khz|mhz|ghz|"
    r"b|kb|mb|gb|tb|kib|mib|gib|tib|bytes?|bits?|bps|kbps|mbps|gbps|"
    r"mm|cm|m|km|in|inch|inches|ft|mil|um|µm|nm|"
    r"g|kg|mg|lb|lbs|oz|l|ml|"
    r"rpm|cfm|dbm|db|dba|"
    r"x|times|"
    r"wk|mo|yr|y"
)
_NUM_WITH_UNIT = re.compile(
    r"^\s*(?:(?:approx\.?|approximately|about|~|≈|up to|maximum|max\.?|minimum|min\.?)\s*)?"
    r"([$€£¥]\s*)?([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:" + _UNIT_WORDS + r")?\s*\.?\s*$",
    re.I,
)


def clean_answer(ans: str) -> str:
    """Reduce a model answer to the bare value the grader expects."""
    if ans is None:
        return ""
    a = str(ans).strip()
    a = a.strip("\"'`“”‘’ ").strip()
    # drop a leading label such as "Answer:" / "Value:"
    a = re.sub(r"^(answer|value|result)\s*[:=]\s*", "", a, flags=re.I).strip()
    a = a.rstrip(".;,: ").strip()
    if not a:
        return ""
    low = a.lower()
    if low in {"", "n/a", "na", "none", "null", "unknown", "not found", "not available", "no answer",
               "i don't know", "i do not know", "cannot be determined", "not specified", "not in corpus",
               "not in the corpus", "insufficient information"}:
        return ""
    if re.match(r"^(not\s+(found|available|specified|stated|provided|present|in|mentioned)|no\s+(answer|information|data|value|match)|"
                r"cannot\s+(be\s+)?(find|determine|answer)|could\s+not|unable\s+to|does\s+not\s+contain|the\s+(corpus|documents?|excerpts?)\s+do(es)?\s+not)",
                low):
        return ""
    if len(a.split()) > 12 or low.startswith(("the ", "it ", "there ", "according ", "based ", "i ", "unable", "this ")):
        # prose. salvage a trailing value after a verb/preposition, otherwise refuse
        m = re.search(r"^.*\b(?:is|was|are|were|in|at|of|to|by|as|equals?|=|:)\s+"
                      r"([A-Za-z0-9$€£¥][^\s,;]*(?:\s+[^\s,;]+){0,2})\s*$", a, re.S)
        if m:
            a = m.group(1).rstrip(".")
        else:
            return ""
    m = _NUM_WITH_UNIT.match(a)
    if m:
        num = m.group(2).replace(",", "")
        return num
    return a


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = t[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
        start = t.find("{", start + 1)
    return {}


_SEP = r"[\s\-\._\u00b7:/#(),]*"


def value_pattern(answer: str) -> "re.Pattern | None":
    """Regex that finds `answer` in raw text as a whole token sequence: pieces of the value may be
    separated by whitespace or punctuation, but the match may not be glued to other letters/digits
    (so '5.0.1' matches '5.0.1' and '5-0-1' but not '15.0.1', and '94' does not match '1994')."""
    pieces = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", answer or "")
    if not pieces:
        return None
    body = _SEP.join(re.escape(p) for p in pieces)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.I)


def text_contains_value(answer: str, text: str) -> bool:
    pat = value_pattern(answer)
    return bool(pat and pat.search(text or ""))


class Deadline:
    def __init__(self, seconds: float):
        self.t0 = time.monotonic()
        self.limit = seconds

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def remaining(self) -> float:
        return self.limit - self.elapsed()

    def ok(self, need: float = 0.0) -> bool:
        return self.remaining() > need
