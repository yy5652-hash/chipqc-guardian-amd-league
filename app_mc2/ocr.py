"""Match 2 OCR pipeline: classify the sign/plate, transcribe, normalise to the grading convention.

Convention (from the official brief):
  * US plates: the registration characters only, never the state name or slogan.
  * Chinese plates: province character + letter + number (e.g. 京A·12345); the interpunct is optional.
  * STOP signs: "STOP". Speed limit: "SPEED LIMIT 65" (the words are on the sign). Advisory plaque: digits only.
  * Multi-line signs: read top to bottom, join lines with single spaces.
Grading normalisation: uppercase, strip whitespace and - . · _ .
"""
from __future__ import annotations

import json
import logging
import re
import time

from PIL import Image, ImageEnhance, ImageOps

from mc3.util import extract_json

log = logging.getLogger("mc2.ocr")

US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA",
    "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE",
    "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS",
    "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA",
    "WASHINGTON DC", "ONTARIO", "QUEBEC", "BRITISH COLUMBIA", "ALBERTA",
}
PLATE_BANNER_WORDS = {
    "THE", "LONE", "STAR", "STATE", "GARDEN", "EMPIRE", "SUNSHINE", "GOLDEN", "EVERGREEN", "FIRST", "IN", "FLIGHT",
    "LIVE", "FREE", "OR", "DIE", "FAMOUS", "POTATOES", "AMERICA'S", "DAIRYLAND", "LAND", "OF", "LINCOLN", "ENCHANTMENT",
    "GREAT", "LAKES", "PURE", "MICHIGAN", "WILD", "WONDERFUL", "VACATIONLAND", "OCEAN", "SPIRIT", "SPORTSMAN'S",
    "PARADISE", "SHOW", "ME", "TREASURE", "BIG", "SKY", "COUNTRY", "VOLUNTEER", "CONSTITUTION", "PEACH",
    "ALOHA", "SCENIC", "WILDLIFE", "ISLAND", "KEYSTONE", "BLUEGRASS", "NATURAL", "GREEN", "MOUNTAIN", "HEART",
    "DIXIE", "BEAUTIFUL", "EXPLORE", "MINNESOTA", "10,000", "LAKES", "MYFLORIDA", "DMV", "GOV", "COM", "WWW",
    "MONTH", "YEAR", "EXP", "EXPIRES", "REGISTRATION", "PLATE", "LICENSE", "LICENCE", "NUMBER", "TAG", "USA",
    "NORTH", "SOUTH", "EAST", "WEST", "NEW", "DC", "CA", "NY", "TX", "FL", "AZ", "NV", "WA", "OR", "CO", "IL",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}
CN_PROVINCE = "京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳台使领学警挂"

SYSTEM = """You are a precision OCR system for road scenes. You will be shown ONE image that contains a vehicle licence plate or a road sign (stop sign, speed limit sign, advisory speed plaque, warning or work-zone sign). Images may be noisy, blurred, dark, glary or photographed at an angle. Read the characters that are physically printed, exactly, in reading order.

Respond with a single JSON object and nothing else:
{"type": "<us_plate|cn_plate|stop|speed_limit|advisory|sign_text|none>", "text": "<transcription>", "lines": ["<line 1>", "<line 2>", ...], "confidence": <0-1>}

Rules by type:
- us_plate: "text" is ONLY the registration characters (letters and digits) in order, with a single space where the plate shows a gap. Do NOT include the state or province name, slogans, county, month/year stickers, "DMV" or website text.
- cn_plate: "text" is the full registration INCLUDING the leading province character and letter, e.g. 京A·12345 or 沪B·88888. Keep the interpunct if present.
- stop: "text" is exactly STOP.
- speed_limit: "text" is the words printed on the sign plus the number, e.g. SPEED LIMIT 65 (or MAXIMUM 50 if that is what is printed).
- advisory: a yellow/orange plaque or sign carrying a number only (no words): "text" is the digits only, e.g. 35.
- sign_text: any other sign with words (work zone, warning, detour, road closed, etc.): "lines" are the printed lines top to bottom and "text" is those lines joined by single spaces, e.g. ROAD WORK AHEAD.
- none: no plate or sign is legible; "text" is "".

Never add labels, units, punctuation or explanations. Letters are uppercase. Distinguish carefully between 0/O, 1/I, 8/B, 5/S, 2/Z using the font and context."""

LOCATE = """Find the licence plate or road sign in this image. Respond with JSON only:
{"found": true/false, "bbox_2d": [x1, y1, x2, y2], "kind": "<plate|sign>"}
Coordinates are relative, on a 0-1000 scale of the image width and height (origin top-left). The box must tightly enclose the whole plate or sign face."""


def prep(img: Image.Image, max_side: int = 1792) -> Image.Image:
    from mc3.llm import to_rgb

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = to_rgb(img)  # 16-bit / float TIFFs would otherwise clip to white
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
    if min(img.size) < 224:  # tiny crops: upsample so the patch grid has enough resolution
        f = 224 / min(img.size)
        img = img.resize((int(img.size[0] * f), int(img.size[1] * f)), Image.LANCZOS)
    return img


def enhance(img: Image.Image) -> Image.Image:
    g = ImageOps.autocontrast(img, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.4)
    g = ImageEnhance.Sharpness(g).enhance(1.6)
    return g


def norm(s: str) -> str:
    return re.sub(r"[\s\-\.·_]+", "", (s or "")).upper()


def clean_us_plate(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9 ]+", " ", text or "").upper()
    tokens = [x for x in t.split() if x]
    # drop state names (possibly multi-word) and slogan words
    joined = " ".join(tokens)
    for st in sorted(US_STATES, key=len, reverse=True):
        joined = re.sub(r"\b" + re.escape(st) + r"\b", " ", joined)
    tokens = [x for x in joined.split() if x and x not in PLATE_BANNER_WORDS]
    # a plate is usually one token of 5-8 chars or two tokens (letters + digits)
    cands = [x for x in tokens if 2 <= len(x) <= 8 and re.search(r"\d", x) or (len(x) >= 2 and x.isalpha() and len(x) <= 4)]
    if not cands:
        cands = tokens
    # prefer the sequence of tokens that mixes letters and digits
    best = " ".join(cands[:2]) if len(cands) >= 2 and sum(len(x) for x in cands[:2]) <= 8 else (cands[0] if cands else "")
    if len(norm(best)) < 4 and tokens:
        best = max(tokens, key=len)
    return best.strip()


def clean_cn_plate(text: str) -> str:
    t = (text or "").strip().upper().replace(" ", "")
    m = re.search(r"([" + CN_PROVINCE + r"])\s*·?\s*([A-Z])\s*·?\s*([A-Z0-9]{4,6})", t)
    if m:
        return f"{m.group(1)}{m.group(2)}·{m.group(3)}"
    return t


def clean_sign(text: str, lines: list[str]) -> str:
    src = " ".join(lines) if lines else (text or "")
    src = re.sub(r"[^A-Za-z0-9 ]+", " ", src).upper()
    return " ".join(src.split())


def postprocess(obj: dict) -> tuple[str, str, float]:
    kind = str(obj.get("type", "")).lower().strip()
    text = str(obj.get("text", "") or "")
    lines = obj.get("lines") or []
    if isinstance(lines, str):
        lines = [lines]
    try:
        conf = float(obj.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    if kind == "us_plate":
        out = clean_us_plate(text)
    elif kind == "cn_plate":
        out = clean_cn_plate(text)
    elif kind == "stop":
        out = "STOP"
    elif kind == "speed_limit":
        s = clean_sign(text, [])
        m = re.search(r"(\d{1,3})", s)
        if "SPEED" in s and "LIMIT" in s and m:
            out = f"SPEED LIMIT {m.group(1)}"
        elif m and "MAXIMUM" in s:
            out = f"MAXIMUM {m.group(1)}"
        else:
            out = s
    elif kind == "advisory":
        m = re.search(r"\d{1,3}", text) or re.search(r"\d{1,3}", " ".join(map(str, lines)))
        out = m.group(0) if m else clean_sign(text, lines)
    elif kind == "sign_text":
        out = clean_sign(text, [str(x) for x in lines])
    elif kind == "none":
        out = ""
    else:
        out = clean_sign(text, [str(x) for x in lines]) if lines else (text or "").strip()
    return out, kind, max(0.0, min(1.0, conf))


class OCR:
    def __init__(self, vlm):
        self.vlm = vlm

    def _ask(self, img: Image.Image, hint: str = "") -> dict:
        parts = [{"type": "image", "image": img}, {"type": "text", "text": (hint + "\n" if hint else "") + "Transcribe. JSON:"}]
        t0 = time.monotonic()
        raw = self.vlm.chat(SYSTEM, parts, max_new_tokens=96)
        obj = extract_json(raw)
        log.info("ocr %.1fs -> %s", time.monotonic() - t0, raw[:200].replace("\n", " "))
        return obj

    def _locate(self, img: Image.Image) -> Image.Image | None:
        try:
            raw = self.vlm.chat(None, [{"type": "image", "image": img}, {"type": "text", "text": LOCATE}], max_new_tokens=64)
            obj = extract_json(raw)
            box_raw = obj.get("bbox_2d") or obj.get("bbox")
            if isinstance(box_raw, list) and box_raw and isinstance(box_raw[0], list):
                box_raw = box_raw[0]
            if obj.get("found") is False or not box_raw or len(box_raw) < 4:
                return None
            x1, y1, x2, y2 = [float(v) for v in box_raw][:4]
            w, h = img.size
            if max(x1, y1, x2, y2) <= 1.0:      # fractions
                x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            elif max(x1, y1, x2, y2) <= 1000:   # Qwen3-VL grounding: relative 0-1000 scale
                x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
            # else: absolute pixels (already in image coordinates)
            x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
            bw, bh = x2 - x1, y2 - y1
            if bw < 16 or bh < 8 or bw > 0.98 * w:
                return None
            pad_x, pad_y = bw * 0.15, bh * 0.25
            box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y)))
            crop = img.crop(box)
            return prep(crop, max_side=1024)
        except Exception as e:  # noqa: BLE001
            log.warning("locate failed: %s", e)
            return None

    def read(self, path: str, deadline_s: float = 20.0) -> dict:
        t0 = time.monotonic()
        img = prep(Image.open(path))
        votes: list[tuple[str, str, float, str]] = []

        def run(im: Image.Image, label: str, hint: str = ""):
            obj = self._ask(im, hint)
            text, kind, conf = postprocess(obj)
            votes.append((text, kind, conf, label))
            log.info("pass %s: type=%s text=%r conf=%.2f", label, kind, text, conf)

        run(img, "full")
        # second opinion on a tight crop when there is time (helps small plates and angled shots)
        if time.monotonic() - t0 < deadline_s * 0.45:
            crop = self._locate(img)
            if crop is not None and time.monotonic() - t0 < deadline_s * 0.6:
                run(crop, "crop")
        # third opinion on an enhanced image if the first two disagree or are weak
        texts = {norm(v[0]) for v in votes if v[0]}
        weak = not votes or not votes[0][0] or votes[0][2] < 0.75
        if (len(texts) > 1 or weak) and time.monotonic() - t0 < deadline_s * 0.75:
            run(enhance(img), "enhanced")
        # decide: majority on the normalised string, ties -> highest confidence, prefer non-empty
        tally: dict[str, list] = {}
        for text, kind, conf, label in votes:
            if not text:
                continue
            tally.setdefault(norm(text), []).append((conf, text, kind, label))
        if not tally:
            return {"text": "", "confidence": 0.0, "passes": [v[3] for v in votes]}
        best_key = max(tally, key=lambda k: (len(tally[k]), max(c for c, *_ in tally[k])))
        conf, text, kind, label = max(tally[best_key])
        agreement = len(tally[best_key]) / max(1, len([v for v in votes if v[0]]))
        return {"text": text, "confidence": round(min(1.0, 0.5 * conf + 0.5 * agreement), 3), "type": kind,
                "passes": [f"{v[3]}:{v[0]}" for v in votes], "seconds": round(time.monotonic() - t0, 2)}
