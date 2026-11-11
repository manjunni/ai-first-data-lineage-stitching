#!/usr/bin/env python3
"""
PLAN B: run the real FastAPI backend with the deterministic LLM stand-in.

Use it for rehearsals, or if Azure OpenAI is slow or unreachable on the day.
Everything else (parsers, stitcher, gap detection, scoring, review rules, publish
gating, audit trail) is the real code; only the LLM calls are stubbed.

    python demo/scripts/run_offline_api.py            # serves on http://localhost:8000

Tell the audience when you use it: the LLM fallback and SME answer extraction are canned.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_llm  # noqa: E402

fake_llm.install()

import uvicorn  # noqa: E402

if __name__ == "__main__":
    os.chdir(ROOT)
    print("OFFLINE STUB: LLM calls are answered by demo/scripts/fake_llm.py, not Azure OpenAI.")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
