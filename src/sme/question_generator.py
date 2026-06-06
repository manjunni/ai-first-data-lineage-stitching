from __future__ import annotations

import concurrent.futures
from typing import Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.models import Gap, LineageGraph, LineageEdge
from config.settings import settings

# ---------------------------------------------------------------------------
# Gap-type templates
# Templates are filled with graph context; LLM polishes them into natural
# questions only when the context is rich enough to warrant it.
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    "missing_source": (
        "Where does the data in '{node_name}' ({system}) come from? "
        "Please provide the source system (e.g. ADLS path, SQL table, Databricks table) "
        "that feeds into this table or file."
    ),
    "missing_target": (
        "Does '{node_name}' ({system}) feed data into any downstream system? "
        "(e.g. Power BI report, SQL Server table, another Databricks table) "
        "If so, what is the target, and how is the data loaded there?"
    ),
    "unknown_system": (
        "Which platform hosts '{node_name}' (documented path: {full_path})? "
        "Is it Azure Data Lake Storage, Databricks, Azure SQL, Power BI, "
        "or another system?"
    ),
    "unresolved_dataset_ref": (
        "The ADF pipeline references a dataset named '{dataset_name}'. "
        "What is its actual storage location? "
        "For example: an ADLS path (abfss://...), a SQL table (schema.table), "
        "or a Databricks Delta table (catalog.schema.table)."
    ),
    "low_confidence_edge": (
        "Can you confirm that data flows from '{source_name}' → '{target_name}'? "
        "The automated analysis found this link but was only {confidence:.0%} confident. "
        "Documented transformation: {transformation}. "
        "Is this correct? If not, please describe the actual relationship."
    ),
    "isolated_subgraph": (
        "These connected tables have no documented data origin: {node_names}. "
        "What system provides the raw data for this group? "
        "Please describe the original source (system type and path or table name)."
    ),
    "contradictory_paths": (
        "There are conflicting records about how data flows from "
        "'{source_name}' to '{target_name}'. "
        "Which of these transformations is correct: {transforms}? "
        "Or please describe the actual transformation."
    ),
}

_LLM_SYSTEM_PROMPT = """\
You are a data governance assistant helping document enterprise data lineage.
You will receive a draft question for a subject matter expert (SME) and supporting context.
Rewrite the question to be clear, concise, and professional — suitable for a business analyst.
- Use plain English, avoid technical jargon where possible.
- Keep the answer to a single question (or numbered sub-questions if unavoidable).
- Do not add preamble or explanation — return ONLY the rewritten question text."""


def generate_questions(
    gaps: list[Gap],
    graph: LineageGraph,
    enrich_with_llm: bool = True,
) -> list[dict]:
    """
    Build one question dict per open gap.

    Each dict contains:
      gap_id, gap_type, question, sme_email, answered, context_summary

    If enrich_with_llm=True, all LLM calls run in parallel (ThreadPoolExecutor).
    If the LLM call fails for a gap, the template question is used as fallback.
    """
    open_gaps = [g for g in gaps if not g.resolved]
    if not open_gaps:
        return []

    # Build template questions (fast, no LLM)
    drafts: list[dict] = []
    for gap in open_gaps:
        template_q, context = _fill_template(gap, graph)
        sme_email = _resolve_sme_email(gap, graph)
        drafts.append({
            "gap_id":          gap.id,
            "gap_type":        gap.gap_type,
            "node_id":         gap.node_id,
            "edge_id":         gap.edge_id,
            "question":        template_q,
            "sme_email":       sme_email,
            "answered":        False,
            "context_summary": context,
        })

    if not enrich_with_llm:
        return drafts

    # Enrich all drafts in parallel
    llm = _get_llm()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_llm_enrich, llm, d["question"], d["context_summary"]): i
            for i, d in enumerate(drafts)
        }
        for future, idx in futures.items():
            try:
                enriched = future.result(timeout=30)
                drafts[idx]["question"] = enriched
            except Exception:
                pass  # keep template question as fallback

    return drafts


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def _fill_template(gap: Gap, graph: LineageGraph) -> tuple[str, str]:
    """Return (filled_question, context_summary) for a gap."""
    tpl = _TEMPLATES.get(gap.gap_type, gap.description)
    context = gap.description

    try:
        if gap.gap_type == "missing_source":
            node = graph.nodes.get(gap.node_id or "")
            if node:
                return tpl.format(node_name=node.name, system=node.system.value), context

        elif gap.gap_type == "missing_target":
            node = graph.nodes.get(gap.node_id or "")
            if node:
                return tpl.format(node_name=node.name, system=node.system.value), context

        elif gap.gap_type == "unknown_system":
            node = graph.nodes.get(gap.node_id or "")
            if node:
                return tpl.format(node_name=node.name, full_path=node.full_path), context

        elif gap.gap_type == "unresolved_dataset_ref":
            node = graph.nodes.get(gap.node_id or "")
            if node:
                dataset_name = node.full_path.replace("adf://", "")
                return tpl.format(dataset_name=dataset_name), context

        elif gap.gap_type == "low_confidence_edge":
            edge = _find_edge(gap.edge_id, graph)
            if edge:
                src = graph.nodes.get(edge.source_id)
                tgt = graph.nodes.get(edge.target_id)
                return tpl.format(
                    source_name=src.name if src else "unknown",
                    target_name=tgt.name if tgt else "unknown",
                    confidence=edge.confidence,
                    transformation=edge.transformation or "undocumented",
                ), context

        elif gap.gap_type == "isolated_subgraph":
            names = _extract_names_from_description(gap.description)
            return tpl.format(node_names=names), context

        elif gap.gap_type == "contradictory_paths":
            edge = _find_edge(gap.edge_id, graph)
            if edge:
                src = graph.nodes.get(edge.source_id)
                tgt = graph.nodes.get(edge.target_id)
                transforms = _extract_transforms_from_description(gap.description)
                return tpl.format(
                    source_name=src.name if src else "unknown",
                    target_name=tgt.name if tgt else "unknown",
                    transforms=transforms,
                ), context

    except (KeyError, AttributeError):
        pass

    return gap.description, context


def _resolve_sme_email(gap: Gap, graph: LineageGraph) -> Optional[str]:
    """Find the best SME email: gap override → node owner → settings default."""
    if gap.sme_email:
        return gap.sme_email
    node = graph.nodes.get(gap.node_id or "")
    if node and node.owner and "@" in node.owner:
        return node.owner
    return getattr(settings, "default_sme_email", None)


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

def _llm_enrich(llm: AzureChatOpenAI, draft: str, context: str) -> str:
    response = llm.invoke([
        SystemMessage(content=_LLM_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nDraft question:\n{draft}"),
    ])
    return response.content.strip()


def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment,
        api_version="2024-08-01-preview",
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_edge(edge_id: Optional[str], graph: LineageGraph) -> Optional[LineageEdge]:
    if not edge_id:
        return None
    return next((e for e in graph.edges if e.id == edge_id), None)


def _extract_names_from_description(desc: str) -> str:
    """Pull the quoted names list out of an isolated_subgraph description."""
    import re
    m = re.search(r"\(([^)]+)\)", desc)
    return m.group(1) if m else desc[:80]


def _extract_transforms_from_description(desc: str) -> str:
    """Pull the transforms list out of a contradictory_paths description."""
    import re
    m = re.search(r"\[([^\]]+)\]", desc)
    return m.group(1) if m else "see description"
