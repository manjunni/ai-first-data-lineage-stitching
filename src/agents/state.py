from __future__ import annotations
from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator

from src.graph.models import LineageFragment, LineageGraph, Gap


def _merge_dict(a: dict, b: dict) -> dict:
    return {**a, **b}


class LineageState(TypedDict):
    run_id: str

    # Uploaded artifact paths/content keyed by filename
    artifacts: list[dict]

    # Parsed fragments from each artifact (accumulated across iterations)
    fragments: Annotated[list[LineageFragment], operator.add]

    # Stitched lineage graph (replaced each iteration)
    lineage_graph: Optional[LineageGraph]

    # Gaps detected in the current graph
    gaps: list[Gap]

    # Questions sent to SMEs this run
    sme_questions: Annotated[list[dict], operator.add]

    # Responses received from SMEs
    sme_responses: list[dict]

    # How many SME iteration loops have run
    iteration: int

    # Per-edge confidence scores after finalization
    confidence_scores: Annotated[dict, _merge_dict]

    # Generated diagram output (Mermaid string, Graphviz dot)
    visuals: Optional[dict]

    # Full audit log (every decision, SME interaction, score change)
    audit_trail: Annotated[list[dict], operator.add]

    # Which catalog to publish to: "alation" | "collibra" | "dry_run"
    target_catalog: str

    # Final status of the run
    status: str

    # Whether lineage has been pushed to catalog
    catalog_published: bool
