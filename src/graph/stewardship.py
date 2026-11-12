"""
External producers and stewardship.

Landing-zone data is often produced outside the data platform: SAP CDC, a WMS
Event Hubs capture, a partner SFTP drop. Nothing in the code base produces it,
so lineage can only start there if someone *declares* the producer:

* an ADF dataset annotation ``declared_producer:<who>``;
* a Unity Catalog table comment that says the data is loaded by someone
  "external" to the platform.

Declared producers become ``SystemType.EXTERNAL`` nodes with an edge into the
dataset. Landing data with neither code nor a declaration is "stewardship
required": it is published that way and routed to a suggested steward.
"""
from __future__ import annotations

import re

from src.graph.models import LineageEdge, LineageGraph, LineageNode, SystemType, ConfidenceLevel

EXT_NS = "ext://"

# Systems where lineage legitimately starts (no upstream required).
ROOT_SYSTEMS = {SystemType.ADLS, SystemType.EXTERNAL}

# "uploaded by Category Buying", "Loaded by Group Treasury FX feed (Logic App)"
_BY_RE = re.compile(
    r"\b(?:uploaded|loaded|maintained|produced|provided|delivered|supplied|dropped|owned|shared|fed)\s+by\s+"
    r"(?:the\s+)?([^.;,]+?(?:\([^)]*\))?)\s*(?:[.;,]|$)",
    re.IGNORECASE,
)
_EXTERNAL_RE = re.compile(r"\bexternal\b", re.IGNORECASE)


def external_path(producer: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", producer.lower()).strip("-")
    return f"{EXT_NS}{slug or 'unknown'}"


def make_external_node(producer: str, description: str | None = None) -> LineageNode:
    return LineageNode(
        name=producer.strip(),
        system=SystemType.EXTERNAL,
        full_path=external_path(producer),
        description=description,
    )


def declared_edge(src: LineageNode, tgt: LineageNode, evidence: str) -> LineageEdge:
    return LineageEdge(
        source_id=src.id,
        target_id=tgt.id,
        transformation="declared external producer",
        confidence=0.80,
        confidence_level=ConfidenceLevel.MEDIUM,
        evidence=f"declared producer: {evidence}",
    )


def producer_from_text(text: str | None) -> str | None:
    """'Weekly prices as uploaded by Category Buying.' → 'Category Buying'."""
    if not text:
        return None
    m = _BY_RE.search(text)
    return m.group(1).strip() if m else None


def declared_external_from_comment(comment: str | None) -> str | None:
    """
    Only a comment that explicitly calls the producer external counts as a
    declaration — "Maintained by Master Data team" alone does not (the
    producer may simply be an artifact that has not been uploaded yet).
    """
    if comment and _EXTERNAL_RE.search(comment):
        return producer_from_text(comment)
    return None


def stewardship_required(graph: LineageGraph) -> dict[str, dict]:
    """
    Landing nodes (ADLS) with no incoming lineage at all — no producing code,
    no declared producer, no SME attestation. Returns node_id → info with a
    suggested steward taken from (in order) the node's own description, the
    descriptions of the tables it feeds, then the node owner.
    """
    incoming = {e.target_id for e in graph.edges if e.review_status != "rejected"}
    out: dict[str, dict] = {}
    for nid, node in graph.nodes.items():
        if node.system != SystemType.ADLS or nid in incoming:
            continue
        hint, hint_from = producer_from_text(node.description), node.name
        if not hint:
            for e in graph.edges:
                if e.source_id == nid and e.target_id in graph.nodes:
                    down = graph.nodes[e.target_id]
                    hint = producer_from_text(down.description)
                    if hint:
                        hint_from = down.full_path
                        break
        out[nid] = {
            "suggested_steward": hint or node.owner,
            "steward_evidence": (
                f"description of {hint_from}" if hint else
                ("dataset owner annotation" if node.owner else None)
            ),
            "owner": node.owner,
        }
    return out
