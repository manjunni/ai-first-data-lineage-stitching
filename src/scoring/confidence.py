from __future__ import annotations

from src.graph.models import LineageGraph, LineageEdge, ConfidenceLevel, SystemType

# ---------------------------------------------------------------------------
# Base scores — set by parsers at parse time, but re-evaluated here using
# all available signals (node metadata, SME status, boundary penalties).
# ---------------------------------------------------------------------------

_BASE_BY_EVIDENCE: list[tuple[str, float]] = [
    # Most specific matches first
    ("sme_approved",    0.95),   # SME has explicitly confirmed
    ("sqlglot",         0.90),   # Parsed deterministically from SQL AST
    ("ast",             0.90),   # Parsed deterministically from Python AST
    ("regex",           0.85),   # Regex extraction (notebooks, DLT)
    ("adf pipeline",    0.85),   # ADF JSON activity traversal
    ("dlt sql",         0.90),   # DLT SQL preprocessed through sqlglot
    ("llm",             0.60),   # LLM-inferred (fallback)
    ("fallback",        0.60),   # Explicit fallback path
]

# Adjustments applied on top of the base score
_PENALTY_CROSS_SYSTEM   = -0.05   # edge crosses a system boundary, not yet SME-confirmed
_PENALTY_UNKNOWN_NODE   = -0.10   # source or target has UNKNOWN system type
_PENALTY_ADF_UNRESOLVED = -0.15   # source or target still has adf:// placeholder path

_ADF_NS = "adf://"


def score_graph(graph: LineageGraph) -> dict[str, float]:
    """
    Re-score every edge in the graph using all available signals.
    Updates edge.confidence and edge.confidence_level in place.
    Returns a dict of edge_id → final score.
    """
    scores: dict[str, float] = {}

    for edge in graph.edges:
        score = _score_edge(edge, graph)
        edge.confidence = score
        edge.confidence_level = _to_level(score, edge.sme_approved_by)
        scores[edge.id] = score

    return scores


def graph_confidence_summary(
    graph: LineageGraph,
    scores: dict[str, float],
) -> dict:
    """
    Return aggregate confidence statistics for the full lineage graph.
    Useful for the API status endpoint and audit trail.
    """
    if not scores:
        return {
            "total_edges": 0,
            "avg_confidence": None,
            "by_level": {level.value: 0 for level in ConfidenceLevel},
            "sme_confirmed": 0,
            "cross_system_edges": 0,
            "unresolved_adf_edges": 0,
        }

    values = list(scores.values())
    level_counts: dict[str, int] = {level.value: 0 for level in ConfidenceLevel}
    sme_confirmed = 0
    cross_system = 0
    unresolved_adf = 0
    pending_review = 0
    rejected_edges = 0

    for edge in graph.edges:
        if edge.review_status == "rejected":
            rejected_edges += 1
        elif edge.review_status == "pending":
            pending_review += 1
        level_counts[edge.confidence_level.value] += 1
        if edge.sme_approved_by:
            sme_confirmed += 1
        if edge.system_boundary:
            cross_system += 1
        src = graph.nodes.get(edge.source_id)
        tgt = graph.nodes.get(edge.target_id)
        if (src and src.full_path.startswith(_ADF_NS) and src.system == SystemType.UNKNOWN) or \
           (tgt and tgt.full_path.startswith(_ADF_NS) and tgt.system == SystemType.UNKNOWN):
            unresolved_adf += 1

    return {
        "total_edges":        len(values),
        "avg_confidence":     round(sum(values) / len(values), 3),
        "min_confidence":     round(min(values), 3),
        "max_confidence":     round(max(values), 3),
        "by_level":           level_counts,
        "sme_confirmed":      sme_confirmed,
        "cross_system_edges": cross_system,
        "unresolved_adf_edges": unresolved_adf,
        "pending_review": pending_review,
        "rejected_edges": rejected_edges,
    }


# ---------------------------------------------------------------------------
# Per-edge scoring
# ---------------------------------------------------------------------------

def _score_edge(edge: LineageEdge, graph: LineageGraph) -> float:
    base = _base_from_evidence(edge)
    base = _apply_penalties(base, edge, graph)
    return round(max(0.0, min(1.0, base)), 2)


def _base_from_evidence(edge: LineageEdge) -> float:
    """Map the evidence string to a base confidence score."""
    if edge.sme_approved_by:
        return _BASE_BY_EVIDENCE[0][1]  # 0.95

    ev = (edge.evidence or "").lower()
    for keyword, score in _BASE_BY_EVIDENCE[1:]:
        if keyword in ev:
            return score

    # Fall back to the confidence already set by the parser (e.g. 0.85 from ADF)
    # so we don't accidentally downgrade a well-evidenced edge.
    if edge.confidence > 0.0:
        return edge.confidence

    return 0.20  # no evidence at all


def _apply_penalties(base: float, edge: LineageEdge, graph: LineageGraph) -> float:
    """Apply graph-aware adjustments to the base score."""
    # Cross-system boundary not yet SME-confirmed
    if edge.system_boundary and not edge.sme_approved_by:
        base += _PENALTY_CROSS_SYSTEM

    src = graph.nodes.get(edge.source_id)
    tgt = graph.nodes.get(edge.target_id)

    # Unresolved ADF placeholder on either endpoint — only penalise when the
    # system type is still UNKNOWN (dataset JSON not uploaded).  Nodes that
    # have been resolved to ADF/ADLS/DATABRICKS are no longer "unresolved".
    for node in (src, tgt):
        if node and node.full_path.startswith(_ADF_NS) and node.system == SystemType.UNKNOWN:
            base += _PENALTY_ADF_UNRESOLVED
            break  # apply once even if both ends are unresolved

    # Unknown system type on source or target (after SME confirmation would fix this)
    for node in (src, tgt):
        if node and node.system == SystemType.UNKNOWN:
            base += _PENALTY_UNKNOWN_NODE
            break

    return base


def _to_level(score: float, sme_approved_by: str | None) -> ConfidenceLevel:
    if sme_approved_by:
        return ConfidenceLevel.CONFIRMED
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    if score >= 0.65:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
