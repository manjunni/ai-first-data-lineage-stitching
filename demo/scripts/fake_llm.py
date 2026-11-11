"""
Deterministic stand-in for Azure OpenAI, for REHEARSAL and BACKUP use only.

It lets the demo (and demo/scripts/validate_demo.py) run with no network and no
Azure OpenAI credentials. It answers the four prompts the agent sends:

  1. Parser LLM fallback          -> relationships found in the notebook config
  2. SME question enrichment      -> returns the draft question unchanged
  3. SME answer extraction        -> simple keyword/regex extraction
  4. anything else                -> empty JSON

If you use this in front of stakeholders, say so. It is NOT GPT-4o and it will
not generalise beyond the Harborline demo artifacts.
"""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

_PARSER_MODULES = (
    "src.parsers.sql_parser",
    "src.parsers.notebook_parser",
    "src.parsers.dlt_parser",
    "src.parsers.adf_parser",
)
_SME_MODULES = (
    "src.sme.question_generator",
    "src.sme.response_processor",
)


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text)


class FakeAzureChatOpenAI:
    def __init__(self, *args, **kwargs) -> None:  # signature-compatible, ignores everything
        pass

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        if "parsing a subject matter expert's reply" in system:
            return _resp(json.dumps(_extract_answer(human)))
        if "data governance assistant" in system:
            return _resp(_passthrough_question(human))
        if "data lineage extraction assistant" in system:
            return _resp(json.dumps(_fallback_lineage(human)))
        return _resp("{}")


# ---------------------------------------------------------------------------
# 1. Parser fallback — resolve the CONFIG dict in the finance adjustments notebook
# ---------------------------------------------------------------------------

def _fallback_lineage(text: str) -> dict:
    cfg = dict(re.findall(r'\\?"(\w+_table)\\?"\s*:\s*\\?"([\w.]+)\\?"', text))
    orders, adj, out = cfg.get("orders_table"), cfg.get("adjustments_table"), cfg.get("output_table")
    if not (orders and adj and out):
        return {"relationships": []}

    def node(path: str) -> dict:
        return {"name": path.split(".")[-1], "full_path": path, "system": "databricks"}

    return {"relationships": [
        {"source": node(orders), "target": node(out),
         "transformation": "left join with finance adjustments; adds adjustment_amount to net_amount",
         "column_mappings": [
             {"source_column": "net_amount", "target_column": "adjusted_net_amount",
              "transformation": "net_amount + coalesce(adjustment_amount, 0)"}]},
        {"source": node(adj), "target": node(out),
         "transformation": "left join on order_id; adjustment_amount applied to net_amount",
         "column_mappings": [
             {"source_column": "adjustment_amount", "target_column": "adjusted_net_amount",
              "transformation": "net_amount + coalesce(adjustment_amount, 0)"}]},
    ]}


# ---------------------------------------------------------------------------
# 2. Question enrichment — keep the template wording
# ---------------------------------------------------------------------------

def _passthrough_question(human: str) -> str:
    return human.split("Draft question:\n", 1)[-1].strip()


# ---------------------------------------------------------------------------
# 3. SME answer extraction
# ---------------------------------------------------------------------------

_ABFSS = re.compile(r"abfss://[^\s,;'\"]+")
_SQL_TABLE = re.compile(r"\b(?:dbo|reporting)\.[A-Za-z_]\w*")
_QUOTED = re.compile(r"['\u2018\u2019\"]([^'\u2018\u2019\"]{3,80})['\u2018\u2019\"]")


def _extract_answer(human: str) -> dict:
    gap_type = (re.search(r"Gap type:\s*(\w+)", human) or [None, ""])[1]
    reply = human.split("SME reply:\n", 1)[-1].strip()
    low = reply.lower()

    out = {"answer_type": "no_information", "primary_value": None,
           "system_type": "unknown", "confirmed": None, "notes": reply[:200]}

    if gap_type in ("low_confidence_edge", "contradictory_paths"):
        if re.search(r"\b(no|incorrect|wrong|not correct)\b[,.]", low) and not re.search(r"\b(yes|confirmed)\b", low):
            return {**out, "answer_type": "correction", "confirmed": False}
        if re.search(r"\b(yes|confirmed|correct|that is right)\b", low):
            return {**out, "answer_type": "confirmation", "confirmed": True}
        return out

    m = _ABFSS.search(reply)
    if m:
        return {**out, "answer_type": "source_path" if gap_type == "missing_source" else "target_path",
                "primary_value": m.group(0).rstrip(".)"), "system_type": "adls"}

    if any(k in low for k in ("power bi", "ssrs", "dashboard", "report")):
        q = _QUOTED.search(reply)
        if q:
            return {**out, "answer_type": "target_path", "primary_value": q.group(1), "system_type": "bi"}

    m = _SQL_TABLE.search(reply)
    if m:
        return {**out, "answer_type": "source_path" if gap_type == "missing_source" else "target_path",
                "primary_value": m.group(0), "system_type": "sql"}
    return out


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------

def install() -> None:
    """Set dummy Azure settings and swap AzureChatOpenAI for the stub in every module."""
    import importlib

    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "http://offline.invalid")
    os.environ.setdefault("AZURE_OPENAI_API_KEY", "offline-stub")
    os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "offline-stub")
    for name in _PARSER_MODULES + _SME_MODULES:
        importlib.import_module(name).AzureChatOpenAI = FakeAzureChatOpenAI
