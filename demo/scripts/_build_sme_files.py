"""Writes the per-question SME answer files (txt / docx / xlsx) from the playbook.
Run from the repo root:  python demo/scripts/_build_sme_files.py
"""
import json
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
OUT = DEMO / "sme_answers"
answers = json.loads((OUT / "sme_answer_playbook.json").read_text())["answers"]

for a in answers:
    path = OUT / a["file"]
    if path.suffix == ".txt":
        path.write_text(a["answer"] + "\n", encoding="utf-8")
    elif path.suffix == ".docx":
        from docx import Document

        doc = Document()
        doc.add_heading("Lineage question - reply", level=2)
        doc.add_paragraph(f"From: {a['persona']}")
        doc.add_paragraph(a["answer"])
        doc.save(path)
    elif path.suffix == ".xlsx":
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Reply"
        ws.append(["From", "Answer"])
        ws.append([a["persona"], a["answer"]])
        wb.save(path)
    print("wrote", path.name)
