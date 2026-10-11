"""Qwen3-VL wrapper (text + images) with a stub for plumbing tests."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import config

log = logging.getLogger("mc3.llm")

OCR_PROMPT = (
    "Transcribe every piece of text in this image exactly as written: titles, labels, table cells, pin names and "
    "pin numbers, part numbers, revisions, serial numbers, dates, codes and values. Keep each label attached to its "
    "value (for example 'B14: THERM_ALERT#' or 'REV: C2'). Write tables one row per line with cells separated by "
    "' | '. Do not describe the picture, do not summarise, do not add commentary. Output only the transcription."
)


def load_image(path: str, max_side: int | None = None):
    from PIL import Image, ImageOps

    max_side = max_side or config.IMAGE_MAX_SIDE
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.LANCZOS)
    if min(img.size) < 56:  # tiny images upset the patchifier
        f = 56 / min(img.size)
        img = img.resize((int(img.size[0] * f) + 1, int(img.size[1] * f) + 1), Image.LANCZOS)
    return img


class VLM:
    def __init__(self, path: Path, device: str, dtype: str):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError:  # older/newer naming
            from transformers import AutoModelForImageTextToText as Cls

        t0 = time.monotonic()
        self.torch = torch
        self.device = device
        dt = getattr(torch, dtype) if device != "cpu" else getattr(torch, __import__("os").environ.get("MC3_CPU_DTYPE", "float32"))
        self.processor = AutoProcessor.from_pretrained(str(path))
        # no device_map (would need accelerate): load on CPU then move explicitly to the GPU
        self.model = Cls.from_pretrained(str(path), dtype=dt, low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()
        log.info("VLM loaded from %s on %s in %.1fs", path, device, time.monotonic() - t0)

    def chat(self, system: str | None, user_parts: list[dict], max_new_tokens: int = 160) -> str:
        """user_parts: [{"type":"text","text":...} | {"type":"image","image": PIL.Image}]"""
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": user_parts})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.0)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return text.strip()

    def transcribe(self, image_path: str) -> str:
        img = load_image(image_path)
        return self.chat(None, [{"type": "image", "image": img}, {"type": "text", "text": OCR_PROMPT}],
                         max_new_tokens=700)


class StubVLM:
    """No model: answers empty, transcribes nothing. Only for contract/plumbing tests."""

    def __init__(self, *a, **k):
        self.device = "cpu"

    def chat(self, system, user_parts, max_new_tokens=160) -> str:
        return '{"answer": "", "evidence": [], "lookup": []}'

    def transcribe(self, image_path: str) -> str:
        return ""
