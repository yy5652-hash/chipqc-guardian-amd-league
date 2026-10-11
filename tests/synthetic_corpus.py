#!/usr/bin/env python3
"""Build a deliberately nasty corpus and run ingestion + chunking only (no model).

  python3 tests/synthetic_corpus.py /tmp/nasty
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))


def build(root: Path) -> None:
    import csv
    import docx
    import fitz
    import openpyxl
    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=True)
    (root / "empty_dir").mkdir(exist_ok=True)
    (root / "deep" / "deeper" / "deepest").mkdir(parents=True, exist_ok=True)
    (root / "deep" / "deeper" / "deepest" / "notes.txt").write_text("Vault code is VX-7781. Replacement cadence: 400 hours.\n")
    (root / "zero.txt").write_bytes(b"")
    (root / "locked.txt").write_text("secret 999")
    os.chmod(root / "locked.txt", 0)
    (root / "blob.bin").write_bytes(os.urandom(4096))
    (root / "latin1.txt").write_bytes("Temperatur\xe9 limite: 87 \xb0C\n".encode("latin-1"))
    (root / "utf16.txt").write_bytes("UTF-16 note: baud rate 115200\n".encode("utf-16"))
    (root / "huge_line.log").write_text("x" * 300000 + " CODE-Z9 ends here\n")
    with open(root / "big.log", "w") as fh:
        for i in range(20000):
            fh.write(f"2026-09-{(i % 28) + 1:02d}T00:{i % 60:02d}:00Z node-{i % 7} INFO heartbeat {i}\n")
            if i == 12345:
                fh.write("2026-09-03T01:02:03Z node-3 ERROR E4455: coolant pump stalled, ticket ORR-2201\n")
    with open(root / "semi.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["sku", "name", "price_usd", "moq"])
        w.writerow(["ORR-PSU-9000-A", "PSU 1600W", "412.5", "10"])
        w.writerow(["ORR-PSU-9001-A", "PSU 2000W", "488.50", "5"])
    with open(root / "quoted.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticket", "summary", "fixed_in"])
        w.writerow(["ORR-2201", "pump stalled, see log; \"urgent\"", "5.0.1"])
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Lead times"
    ws.append(["part", "weeks", "note"])
    ws.append(["ORR-PSU-9000-A", 6, "allocation"])
    ws2 = wb.create_sheet("Costs")
    ws2.append(["part", "cost"])
    ws2.append(["ORR-PSU-9001-A", 300])
    ws3 = wb.create_sheet("Blank")
    wb.save(root / "parts.xlsx")
    d = docx.Document()
    d.add_paragraph("Roadmap FY28 — internal")
    t = d.add_table(rows=2, cols=3)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text, t.rows[0].cells[2].text = "Product", "Milestone", "Quarter"
    t.rows[1].cells[0].text, t.rows[1].cells[1].text, t.rows[1].cells[2].text = "TQ-80", "tape-out", "Q1 FY28"
    d.add_paragraph("Nothing else.")
    d.save(root / "roadmap.docx")
    # image with text, and an image-only pdf page + a text pdf + an encrypted pdf
    img = Image.new("RGB", (640, 200), "white")
    ImageDraw.Draw(img).text((20, 80), "BOARD REV: REV-D7   SERIAL SN-00042", fill="black")
    img.save(root / "label.png")
    img2 = Image.new("RGB", (640, 200), "white")
    ImageDraw.Draw(img2).text((20, 80), "ASSET TAG AT-5531   SITE: ROME-2", fill="black")
    img2.save(root / "asset_tag.tif")
    img3 = Image.new("RGB", (640, 200), "white")
    ImageDraw.Draw(img3).text((20, 80), "PAGE 2 LABEL   BOARD REV: REV-E1", fill="black")
    img3.save(root / "_page2.png")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "TQ-80 datasheet r3. Maximum junction temperature 97 C. Supersedes r2.")
    page2 = pdf.new_page()
    page2.insert_image(page2.rect, filename=str(root / "_page2.png"))
    pdf.save(root / "tq80_datasheet_r3.pdf")
    pdf.save(root / "tq80_datasheet_r2_SUPERSEDED.pdf")
    (root / "_page2.png").unlink()
    enc = fitz.open()
    enc.new_page().insert_text((72, 72), "Unit price at 10000 units: 6412")
    enc.save(root / "contract_ENCRYPTED.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="nope", owner_pw="nope2")
    (root / "svc.py").write_text("DEFAULT_RETRIES = 7  # was 3\n\ndef run(timeout_s=240, retries=DEFAULT_RETRIES):\n    return timeout_s\n")
    try:
        os.symlink(root / "missing_target.txt", root / "dangling_link.txt")
    except Exception:
        pass


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    build(root)
    os.environ["MC3_STUB"] = "1"
    from mc3.ingest import ingest_corpus
    from mc3.chunk import build_chunks
    from mc3.util import setup_logging
    setup_logging("mc3")
    t0 = time.monotonic()
    docs = ingest_corpus(root)
    chunks = build_chunks(docs)
    dt = time.monotonic() - t0
    print(f"\n{len(docs)} files, {len(chunks)} chunks in {dt:.1f}s")
    for d in docs:
        print(f"  {d.rel:45} kind={d.kind:6} segs={len(d.segments):3} status={d.status or '-':10} ocr_pages={d.ocr_pages} err={d.error or '-'}")
    must_have = ["semi.csv", "quoted.csv", "parts.xlsx", "roadmap.docx", "tq80_datasheet_r3.pdf", "svc.py", "big.log",
                 "latin1.txt", "utf16.txt", "deep/deeper/deepest/notes.txt", "label.png", "asset_tag.tif", "huge_line.log"]
    ok = True
    by = {d.rel: d for d in docs}
    for rel in must_have:
        d = by.get(rel)
        if not d or (not d.segments and d.kind != "image"):
            print("MISSING/EMPTY:", rel); ok = False
    for rel in ["locked.txt", "contract_ENCRYPTED.pdf", "blob.bin", "zero.txt"]:
        d = by.get(rel)
        if not d or not d.error:
            print("SHOULD HAVE BEEN SKIPPED:", rel); ok = False
    texts = "\n".join(c.text for c in chunks)
    for needle in ["ORR-2201", "E4455", "412.5", "Q1 FY28", "DEFAULT_RETRIES = 7", "timeout_s = 240", "115200", "87", "CODE-Z9", "5.0.1", "ORR-PSU-9001-A"]:
        if needle not in texts:
            print("VALUE NOT INDEXED:", needle); ok = False
    print("SUPERSEDED status:", by["tq80_datasheet_r2_SUPERSEDED.pdf"].status, "| r3 ocr pages:", by["tq80_datasheet_r3.pdf"].ocr_pages)
    print("RESULT:", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
