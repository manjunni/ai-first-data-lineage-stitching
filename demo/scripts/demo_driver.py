#!/usr/bin/env python3
"""
Demo driver for the Harborline net-revenue lineage demo.

It talks to the running FastAPI backend (the same endpoints the Streamlit UI
uses), so you can script the boring parts of the demo and keep the interesting
parts live in the UI.

    python demo/scripts/demo_driver.py start                 # upload the 10 artifacts, start a run
    python demo/scripts/demo_driver.py questions             # show the SME questions + gap ids
    python demo/scripts/demo_driver.py answer                # submit all SME answers (one email-style reply)
    python demo/scripts/demo_driver.py answer --only A1 A2   # submit a subset (keeps the rest for the live UI)
    python demo/scripts/demo_driver.py answer --from-files   # answer from the txt/docx/xlsx files instead of typed text
    python demo/scripts/demo_driver.py approve --exclude-evidence llm sme
                                                            # bulk-approve parser-derived edges as the Data Owner
    python demo/scripts/demo_driver.py status                # run state, gaps, pending reviews
    python demo/scripts/demo_driver.py publish               # dry-run publish (Data Owner approvals required)

Environment: API_BASE_URL (default http://localhost:8000), DEMO_OWNER_USER / DEMO_OWNER_PASSWORD.
The run id is remembered in demo/.demo_run_id.

NOTE: `approve` is a demo accelerator. The agent has no auto-approve feature; this script simply
acts as the Data Owner through the review API. Say so when you use it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = DEMO_DIR / "artifacts"
PLAYBOOK = DEMO_DIR / "sme_answers" / "sme_answer_playbook.json"
RUN_FILE = DEMO_DIR / ".demo_run_id"


class Api:
    """Thin wrapper that works with requests.Session (live) or a FastAPI TestClient (validation)."""

    def __init__(self, session, base: str = "") -> None:
        self.s, self.base = session, base.rstrip("/")

    def _check(self, r):
        if r.status_code >= 400:
            raise SystemExit(f"API {r.status_code}: {r.text[:400]}")
        return r.json()

    def get(self, path, **kw):
        return self._check(self.s.get(self.base + path, **kw))

    def post(self, path, **kw):
        return self._check(self.s.post(self.base + path, **kw))

    def post_raw(self, path, **kw):
        return self.s.post(self.base + path, **kw)


def artifact_files() -> list[Path]:
    files = sorted(p for p in ARTIFACT_DIR.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No artifacts found in {ARTIFACT_DIR}")
    return files


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_start(api: Api, run_file: Path = RUN_FILE, catalog: str = "dry_run") -> str:
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in artifact_files()]
    run = api.post("/runs", files=files, data={"catalog": catalog})
    run_file.write_text(run["run_id"])
    print(f"Run {run['run_id']}")
    print(f"  status={run['status']}  nodes={run['node_count']}  edges={run['edge_count']}  "
          f"open gaps={run['gaps_open']}  questions pending={run['sme_questions_pending']}")
    return run["run_id"]


def _run_id(run_file: Path = RUN_FILE) -> str:
    if not run_file.exists():
        raise SystemExit("No run yet. Run:  demo_driver.py start")
    return run_file.read_text().strip()


def cmd_questions(api: Api, run_id: str, show: bool = True) -> list[dict]:
    qs = api.get(f"/runs/{run_id}/questions")["questions"]
    if show:
        for i, q in enumerate(qs, 1):
            print(f"Q{i} [{q['gap_type']}] gap_id={q['gap_id']}\n   {q['question']}\n")
    return qs


def load_playbook() -> list[dict]:
    return json.loads(PLAYBOOK.read_text())["answers"]


def match_answers(questions: list[dict], only: list[str] | None = None) -> list[tuple[dict, dict]]:
    """Pair each playbook answer with the question it belongs to (gap_type + keywords)."""
    pairs, used = [], set()
    for ans in load_playbook():
        if only and ans["id"] not in only:
            continue
        for q in questions:
            if q["gap_id"] in used or q["gap_type"] != ans["gap_type"]:
                continue
            text = q["question"].lower()
            if all(k.lower() in text for k in ans["match"]):
                pairs.append((ans, q))
                used.add(q["gap_id"])
                break
    return pairs


def answer_text(a: dict, from_files: bool) -> str:
    """Typed answer, or the text extracted from the answer file (what the UI's per-question uploader does)."""
    if not from_files:
        return a["answer"]
    from src.sme.file_handlers import extract_text_from_file

    f = DEMO_DIR / "sme_answers" / a["file"]
    return extract_text_from_file(f.name, f.read_bytes()).strip()


def cmd_answer(api: Api, run_id: str, sme_email: str, only: list[str] | None = None,
               from_files: bool = False) -> dict:
    questions = cmd_questions(api, run_id, show=False)
    pairs = match_answers(questions, only)
    if not pairs:
        raise SystemExit("No playbook answers matched the current questions.")
    body = "\n".join(f"[REF:{q['gap_id']}]: {answer_text(a, from_files)}" for a, q in pairs)
    from src.sme.email_handler import parse_email_responses  # same parser the UI uses

    parsed = parse_email_responses([{"from": sme_email,
                                     "subject": f"[Lineage Agent] Run {run_id} - reply",
                                     "body": body}])
    for p in parsed:
        p["run_id"] = run_id
    run = api.post(f"/runs/{run_id}/responses", json={"responses": parsed})
    print(f"Submitted {len(parsed)} answers ({', '.join(a['id'] for a, _ in pairs)}).")
    print(f"  status={run['status']}  nodes={run['node_count']}  edges={run['edge_count']}  "
          f"open gaps={run['gaps_open']}  questions pending={run['sme_questions_pending']}")
    return run


def cmd_approve(api: Api, run_id: str, user: str, password: str, exclude_evidence: list[str],
                min_confidence: float, only_pending: bool = True) -> int:
    login = api.post("/auth/login", json={"username": user, "password": password})
    role = login["role"]
    edges = api.get(f"/runs/{run_id}/lineage")["edges"]
    approved = skipped = 0
    for e in edges:
        evidence = (e.get("evidence") or "").lower()
        if only_pending and e["review_status"] != "pending":
            continue
        if any(x.lower() in evidence for x in exclude_evidence) or e["confidence"] < min_confidence:
            skipped += 1
            continue
        api.post(f"/runs/{run_id}/lineage/edges/{e['id']}/review",
                 json={"decision": "approve", "reviewer": user, "role": role,
                       "reason": "Demo bulk approval of deterministic parser output"})
        approved += 1
    print(f"Approved {approved} edge(s) as {user} ({role}); left {skipped} for live review.")
    return approved


def cmd_status(api: Api, run_id: str) -> dict:
    run = api.get(f"/runs/{run_id}")
    gaps = api.get(f"/runs/{run_id}/gaps")
    lineage = api.get(f"/runs/{run_id}/lineage")
    pending = sum(1 for e in lineage["edges"] if e["review_status"] != "approved")
    print(f"Run {run_id}\n  status={run['status']}  iteration={run['iteration']}")
    print(f"  nodes={run['node_count']}  edges={run['edge_count']}  gaps open/total={gaps['open']}/{gaps['total']}")
    print(f"  edges not yet approved: {pending}")
    for g in gaps["gaps"]:
        if not g["resolved"]:
            print(f"    OPEN [{g['gap_type']}] {g['description'][:100]}")
    return {"run": run, "gaps": gaps, "pending": pending}


def cmd_publish(api: Api, run_id: str, catalog: str = "dry_run") -> dict:
    res = api.post(f"/runs/{run_id}/publish", json={"catalog": catalog})
    print(f"Publish: success={res['success']} status={res['status']} "
          f"nodes={res['nodes_published']} edges={res['edges_published']} errors={res['errors']}")
    return res


# ---------------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, str(DEMO_DIR.parent))  # allow `from src...` when run from anywhere
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["start", "questions", "answer", "approve", "status", "publish"])
    ap.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--only", nargs="*", help="answer: playbook ids to submit, e.g. A1 A2")
    ap.add_argument("--from-files", action="store_true",
                    help="answer: use the text extracted from demo/sme_answers/* (txt, docx, xlsx) instead of typed text")
    ap.add_argument("--sme", default=os.getenv("DEMO_SME_USER", "marcus.oyelaran@harborline.example"))
    ap.add_argument("--user", default=os.getenv("DEMO_OWNER_USER", "priya.nair@harborline.example"))
    ap.add_argument("--password", default=os.getenv("DEMO_OWNER_PASSWORD", ""))
    ap.add_argument("--exclude-evidence", nargs="*", default=["llm", "sme"],
                    help="approve: skip edges whose evidence contains any of these words")
    ap.add_argument("--min-confidence", type=float, default=0.80)
    ap.add_argument("--catalog", default="dry_run")
    a = ap.parse_args()

    import requests

    api = Api(requests.Session(), a.base_url)
    if a.command == "start":
        cmd_start(api, catalog=a.catalog)
        return
    rid = _run_id()
    if a.command == "questions":
        cmd_questions(api, rid)
    elif a.command == "answer":
        cmd_answer(api, rid, a.sme, a.only, a.from_files)
    elif a.command == "approve":
        if not a.password:
            raise SystemExit("Set DEMO_OWNER_PASSWORD or pass --password")
        cmd_approve(api, rid, a.user, a.password, a.exclude_evidence, a.min_confidence)
    elif a.command == "status":
        cmd_status(api, rid)
    elif a.command == "publish":
        cmd_publish(api, rid, a.catalog)


if __name__ == "__main__":
    main()
