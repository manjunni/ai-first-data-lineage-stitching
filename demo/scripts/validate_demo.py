#!/usr/bin/env python3
"""
Headless end-to-end check of the demo pack against the REAL agent code.

Runs the full journey in-process (FastAPI TestClient, LangGraph agent, real parsers/stitcher/
gap detector/scoring/review/publish rules) with the deterministic LLM stand-in, and asserts
the outcomes the demo script promises. Run it before every demo:

    python demo/scripts/validate_demo.py            # from the repo root
    python demo/scripts/validate_demo.py --snapshot # also refresh demo/expected/*

Exit code 0 = the pack behaves as documented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

USERS = {
    "priya.nair@harborline.example": {"password": "demo-owner", "role": "Data Owner"},
    "tomasz.kowalski@harborline.example": {"password": "demo-steward", "role": "Domain Steward"},
    "marcus.oyelaran@harborline.example": {"password": "demo-sme", "role": "SME"},
    "chloe.park@harborline.example": {"password": "demo-analyst", "role": "Analyst"},
}
os.environ["LINEAGE_USERS_JSON"] = json.dumps(USERS)

import fake_llm  # noqa: E402

fake_llm.install()

from fastapi.testclient import TestClient  # noqa: E402

import demo_driver as dd  # noqa: E402
from src.api.main import app  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="write demo/expected/* from this run")
    args = ap.parse_args()

    client = TestClient(app)
    api = dd.Api(client)
    run_file = Path(tempfile.mkdtemp()) / "run_id"
    owner, steward, sme, analyst = list(USERS)

    print("\n1. Start run (10 artifacts, 4 parser types)")
    run_id = dd.cmd_start(api, run_file)
    run = api.get(f"/runs/{run_id}")
    check("run pauses for SME input", run["status"] == "emails_sent", run["status"])
    lineage = api.get(f"/runs/{run_id}/lineage")
    names = {n["full_path"]: n for n in lineage["nodes"]}
    check("ADF dataset merged into Databricks table node (bronze orders)",
          "harborline.bronze.orders_raw" in names and not any(p.startswith("adf://bronze_orders") for p in names))
    check("ADF dataset merged into SQL table node (dbo.orders)",
          "dbo.orders" in names and "adf://salesdb_orders" not in names)
    check("legacy SQL + ADF + notebook + DLT all contributed nodes",
          {"reporting.daily_sales_summary", "silver_order_enriched",
           "harborline.silver.orders_clean", "adf://pos_export_orders"} <= set(names))
    adjusted_edges = [e for e in lineage["edges"] if "LLM fallback" in (e["evidence"] or "")]
    check("LLM fallback produced the 2 finance-adjustment edges", len(adjusted_edges) == 2, str(len(adjusted_edges)))
    check("LLM-derived edges are scored below parser edges",
          all(e["confidence"] <= 0.60 for e in adjusted_edges))
    boundaries = {e["system_boundary"] for e in lineage["edges"] if e["system_boundary"]}
    check("cross-system boundaries annotated", {"adls→sql", "sql→databricks", "databricks→dlt"} <= boundaries,
          str(boundaries))
    col_edges = [e for e in lineage["edges"] if e["column_mappings"]]
    check("column-level lineage present", len(col_edges) >= 4, str(len(col_edges)))

    lineage_pre = lineage
    print("\n2. Gaps and SME questions")
    gaps = api.get(f"/runs/{run_id}/gaps")
    by_type: dict[str, int] = {}
    for g in gaps["gaps"]:
        by_type[g["gap_type"]] = by_type.get(g["gap_type"], 0) + 1
    print("     gaps:", by_type)
    check("7 gaps: 3 missing_source, 2 missing_target, 2 low_confidence_edge",
          by_type == {"missing_source": 3, "missing_target": 2, "low_confidence_edge": 2}, str(by_type))
    questions = dd.cmd_questions(api, run_id, show=False)
    pairs = dd.match_answers(questions)
    check("every question has a playbook answer", len(pairs) == len(questions) == 7,
          f"{len(pairs)} matched of {len(questions)}")

    print("\n3. Review rules and publish gating (before any approval)")
    r = client.post(f"/runs/{run_id}/publish", json={"catalog": "dry_run"})
    check("publish refused while run is waiting for SME (409)", r.status_code == 409, str(r.status_code))
    api.post("/auth/login", json={"username": analyst, "password": "demo-analyst"})
    bad = client.post("/auth/login", json={"username": owner, "password": "wrong"})
    check("bad password rejected (401)", bad.status_code == 401)

    print("\n4. SME answers (typed, as one email-style reply)")
    run = dd.cmd_answer(api, run_id, sme)
    check("run finalises after answers", run["status"] in ("visuals_ready", "finalized"), run["status"])
    check("no open gaps remain", run["gaps_open"] == 0, str(run["gaps_open"]))
    lineage = api.get(f"/runs/{run_id}/lineage")
    lineage_post = lineage
    names = {n["full_path"]: n for n in lineage["nodes"]}
    check("SME-provided ADLS sources added",
          "abfss://landing@harborlake.dfs.core.windows.net/returns" in names
          and "abfss://finance@harborlake.dfs.core.windows.net/gl_adjustments" in names)
    check("SME answer reused existing dbo.customers node (no duplicate)",
          sum(1 for n in lineage["nodes"] if n["full_path"].endswith("dbo.customers")) == 1)
    check("Power BI and SSRS targets added as BI nodes",
          {n["name"] for n in lineage["nodes"] if n["system"] == "bi"} == {"Executive Net Revenue", "Monthly Revenue Pack"},
          str({n["name"] for n in lineage["nodes"] if n["system"] == "bi"}))
    check("SME-confirmed edges are CONFIRMED (0.95)",
          sum(1 for e in lineage["edges"] if e["confidence_level"] == "confirmed") >= 6)

    print("\n5. Data Owner review and publish gating")
    r = client.post(f"/runs/{run_id}/publish", json={"catalog": "dry_run"})
    check("publish blocked until Data Owner approves every edge (403)", r.status_code == 403, str(r.status_code))
    edge = lineage["edges"][0]
    r = client.post(f"/runs/{run_id}/lineage/edges/{edge['id']}/review",
                    json={"decision": "approve", "reviewer": analyst, "role": "Analyst"})
    check("analyst can record an initial decision", r.status_code == 200, r.text[:100])
    r = client.post(f"/runs/{run_id}/lineage/edges/{edge['id']}/review",
                    json={"decision": "reject", "reviewer": analyst, "role": "Data Owner"})
    check("role spoofing rejected (403)", r.status_code == 403, str(r.status_code))
    r = client.post(f"/runs/{run_id}/lineage/edges/{edge['id']}/review",
                    json={"decision": "reject", "reviewer": owner, "role": "Data Owner", "reason": ""})
    check("rejection without a reason refused (422)", r.status_code == 422, str(r.status_code))
    n_bulk = dd.cmd_approve(api, run_id, owner, "demo-owner", ["llm", "sme"], 0.80)
    pending = [e for e in api.get(f"/runs/{run_id}/lineage")["edges"] if e["review_status"] != "approved"]
    print(f"     bulk-approved {n_bulk}; left for live review: {len(pending)}")
    check("LLM/SME-derived edges are left for live review", 6 <= len(pending) <= 8, str(len(pending)))
    r = client.post(f"/runs/{run_id}/publish", json={"catalog": "dry_run"})
    check("publish still blocked with edges pending (403)", r.status_code == 403, str(r.status_code))
    for e in pending:
        api.post(f"/runs/{run_id}/lineage/edges/{e['id']}/review",
                 json={"decision": "approve", "reviewer": owner, "role": "Data Owner", "reason": "validated"})
    r = client.post(f"/runs/{run_id}/publish", json={"catalog": "dry_run"})
    check("an Analyst-approved edge does not satisfy publish gating (403)", r.status_code == 403, str(r.status_code))
    # a higher-ranked role may override a lower-ranked decision
    api.post(f"/runs/{run_id}/lineage/edges/{edge['id']}/review",
             json={"decision": "approve", "reviewer": owner, "role": "Data Owner", "reason": "owner override"})
    r = client.post(f"/runs/{run_id}/lineage/edges/{pending[0]['id']}/review",
                    json={"decision": "reject", "reviewer": owner, "role": "Data Owner", "reason": "changed my mind"})
    check("Data Owner approval is read-only afterwards (403)", r.status_code == 403, str(r.status_code))
    res = dd.cmd_publish(api, run_id)
    check("dry-run publish succeeds", res["success"] is True and res["errors"] == [], str(res["errors"]))

    print("\n6. Audit trail")
    audit = api.get(f"/runs/{run_id}/audit")["audit_trail"]
    steps = [a["step"] for a in audit]
    for s in ("ingest", "parse", "stitch", "gap_detection", "question_gen", "send_email", "sme_response",
              "process_responses", "scoring", "visualize", "lineage_review"):
        check(f"audit has '{s}'", s in steps)

    print("\n7. Same journey, SME answers supplied as txt/docx/xlsx files")
    run2 = dd.cmd_start(api, run_file.with_name("run2"))
    r2 = dd.cmd_answer(api, run2, sme, from_files=True)
    check("file-based answers produce the same outcome (18 nodes, 18 edges, 0 open gaps)",
          (r2["node_count"], r2["edge_count"], r2["gaps_open"]) == (18, 18, 0),
          f"{r2['node_count']}/{r2['edge_count']}/{r2['gaps_open']}")

    if args.snapshot:
        out = ROOT / "demo" / "expected"
        out.mkdir(exist_ok=True)
        def shape(g: dict) -> dict:
            nid = {n["id"]: n for n in g["nodes"]}
            return {
                "node_count": len(g["nodes"]), "edge_count": len(g["edges"]),
                "nodes": sorted(({"name": n["name"], "system": n["system"], "path": n["full_path"]}
                                 for n in g["nodes"]), key=lambda x: (x["system"], x["path"])),
                "edges": sorted(({"source": nid[e["source_id"]]["name"], "target": nid[e["target_id"]]["name"],
                                  "transformation": e["transformation"], "boundary": e["system_boundary"],
                                  "evidence": e["evidence"], "confidence": e["confidence"],
                                  "column_mappings": [f"{m['source_column']} -> {m['target_column']}"
                                                      for m in e["column_mappings"]]} for e in g["edges"]),
                                key=lambda x: (x["source"], x["target"])),
            }
        (out / "expected_graph.json").write_text(json.dumps({
            "after_parse_and_stitch": shape(lineage_pre),
            "after_sme_answers": shape(lineage_post),
        }, indent=2, ensure_ascii=False) + "\n")
        (out / "expected_questions.json").write_text(json.dumps(
            [{"id": a["id"], "gap_type": a["gap_type"], "persona": a["persona"], "answer": a["answer"],
              "question_template": q["question"]} for a, q in pairs], indent=2, ensure_ascii=False) + "\n")
        (out / "expected_mermaid_after_sme.mmd").write_text(api.get(f"/runs/{run_id}/visuals")["mermaid"] + "\n")
        # "before" picture: a fresh run that is still waiting for the SME, gaps highlighted in red
        before = dd.cmd_start(api, run_file.with_name("run_before"))
        (out / "expected_mermaid_before_sme.mmd").write_text(api.get(f"/runs/{before}/visuals")["mermaid"] + "\n")
        (out / "expected_publish_payload.json").write_text(
            json.dumps(res.get("payload_summary"), indent=2, ensure_ascii=False)[:200000] + "\n")
        print(f"\n  snapshots written to {out}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
