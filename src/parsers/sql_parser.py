from __future__ import annotations
import json
import uuid

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.parsers.base_parser import BaseParser
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from config.settings import settings

# Dialects tried in order; None = generic fallback
_DIALECTS = ["spark", "tsql", None]

# Path prefixes that indicate an ADLS/cloud storage reference
_ADLS_PREFIXES = ("abfss://", "wasbs://", "adls://", "abfs://")


class SqlParser(BaseParser):
    """
    Parses .sql scripts using sqlglot AST traversal.
    Falls back to LLM when sqlglot cannot parse or yields no lineage.
    """

    def can_parse(self, artifact: dict) -> bool:
        return artifact.get("filename", "").lower().endswith(".sql")

    def parse(self, artifact: dict) -> LineageFragment:
        sql = artifact.get("content", "")
        filename = artifact.get("filename", "unknown.sql")

        try:
            fragment = _parse_with_sqlglot(sql, filename)
            if fragment.edges:
                return fragment
            # sqlglot parsed cleanly but found no lineage — try LLM
        except Exception:
            pass

        return _parse_with_llm(sql, filename)


# ---------------------------------------------------------------------------
# sqlglot path
# ---------------------------------------------------------------------------

def _parse_with_sqlglot(sql: str, filename: str) -> LineageFragment:
    statements = _try_parse(sql)
    if not statements:
        raise ValueError("sqlglot could not parse SQL with any dialect")

    fragment = LineageFragment(artifact_name=filename, artifact_type="sql")
    # Deduplicate nodes within this file by full_path
    node_registry: dict[str, LineageNode] = {}

    for stmt in statements:
        if stmt is None:
            continue
        targets = _extract_targets(stmt)
        sources = _extract_sources(stmt)
        transform = _extract_transformation(stmt)

        for tgt_info in targets:
            tgt_node = _get_or_create_node(tgt_info, node_registry, fragment)
            for src_info in sources:
                src_node = _get_or_create_node(src_info, node_registry, fragment)
                fragment.edges.append(LineageEdge(
                    source_id=src_node.id,
                    target_id=tgt_node.id,
                    transformation=transform,
                    confidence=0.90,
                    confidence_level=ConfidenceLevel.HIGH,
                    evidence=f"parsed from {filename} via sqlglot",
                ))

    return fragment


def _try_parse(sql: str) -> list[exp.Expression] | None:
    for dialect in _DIALECTS:
        try:
            stmts = sqlglot.parse(sql, dialect=dialect, error_level=sqlglot.ErrorLevel.WARN)
            return stmts
        except SqlglotError:
            continue
    return None


def _extract_targets(stmt: exp.Expression) -> list[dict]:
    """Return write-side table info from INSERT, CREATE TABLE/VIEW, MERGE."""
    targets = []

    if isinstance(stmt, exp.Insert):
        if stmt.this:
            targets.append(_table_expr_to_info(stmt.this))

    elif isinstance(stmt, exp.Create):
        obj = stmt.this
        # CREATE TABLE schema.table(...) — obj may be exp.Schema wrapping exp.Table
        table = obj.this if isinstance(obj, exp.Schema) else obj
        if isinstance(table, exp.Table):
            targets.append(_table_expr_to_info(table))

    elif isinstance(stmt, exp.Merge):
        if stmt.this:
            targets.append(_table_expr_to_info(stmt.this))

    return [t for t in targets if t]


def _extract_sources(stmt: exp.Expression) -> list[dict]:
    """Return read-side table info — tables in FROM/JOIN/USING clauses."""
    sources = []

    # For INSERT and CREATE ... AS SELECT, sources live in the SELECT sub-expression.
    # For MERGE, the source is stmt.args["using"].
    if isinstance(stmt, exp.Merge):
        using = stmt.args.get("using")
        if using and isinstance(using, exp.Table):
            sources.append(_table_expr_to_info(using))
        elif using:
            # subquery alias — extract inner tables
            for t in using.find_all(exp.Table):
                sources.append(_table_expr_to_info(t))
        return [s for s in sources if s]

    # Walk all exp.From and exp.Join nodes — this covers nested subqueries too
    target_table_name = _target_name(stmt)
    for table in stmt.find_all(exp.Table):
        # Skip if this table IS the target (present in exp.Insert.this or exp.Create.this)
        if _full_name(table) == target_table_name:
            continue
        info = _table_expr_to_info(table)
        if info:
            sources.append(info)

    # Deduplicate by full_path
    seen: set[str] = set()
    deduped = []
    for s in sources:
        if s["full_path"] not in seen:
            seen.add(s["full_path"])
            deduped.append(s)
    return deduped


def _extract_transformation(stmt: exp.Expression) -> str | None:
    """Summarise what the statement does beyond a direct copy."""
    parts = []

    select = stmt.find(exp.Select)
    if not select:
        return None

    if stmt.find(exp.Where):
        parts.append("filtered")
    if stmt.find(exp.Group):
        parts.append("aggregated")

    join_count = sum(1 for _ in stmt.find_all(exp.Join))
    if join_count:
        parts.append(f"{join_count} join(s)")

    if stmt.find(exp.Window):
        parts.append("windowed")

    # Detect any non-trivial column expressions (not bare column refs or *)
    computed = [
        c for c in select.find_all(exp.Alias)
        if not isinstance(c.this, (exp.Column, exp.Star))
    ]
    if computed:
        parts.append(f"{len(computed)} computed column(s)")

    return ", ".join(parts) if parts else "direct copy"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _table_expr_to_info(table: exp.Table) -> dict | None:
    if not isinstance(table, exp.Table):
        return None
    name = table.name
    db = table.args.get("db")
    catalog = table.args.get("catalog")

    db_str = db.name if db else None
    cat_str = catalog.name if catalog else None

    if cat_str and db_str:
        full_path = f"{cat_str}.{db_str}.{name}"
    elif db_str:
        full_path = f"{db_str}.{name}"
    else:
        full_path = name

    return {"name": name, "full_path": full_path, "system": _infer_system(full_path, cat_str)}


def _get_or_create_node(
    info: dict,
    registry: dict[str, LineageNode],
    fragment: LineageFragment,
) -> LineageNode:
    path = info["full_path"]
    if path in registry:
        return registry[path]
    node = LineageNode(
        id=str(uuid.uuid4()),
        name=info["name"],
        system=info["system"],
        full_path=path,
    )
    registry[path] = node
    fragment.nodes.append(node)
    return node


def _infer_system(full_path: str, catalog: str | None) -> SystemType:
    if any(full_path.startswith(p) for p in _ADLS_PREFIXES):
        return SystemType.ADLS
    # Unity Catalog / Hive Metastore 3-part names → Databricks
    if catalog:
        return SystemType.DATABRICKS
    return SystemType.SQL


def _full_name(table: exp.Table) -> str | None:
    info = _table_expr_to_info(table)
    return info["full_path"] if info else None


def _target_name(stmt: exp.Expression) -> str | None:
    if isinstance(stmt, (exp.Insert, exp.Merge)):
        return _full_name(stmt.this) if isinstance(stmt.this, exp.Table) else None
    if isinstance(stmt, exp.Create):
        obj = stmt.this
        table = obj.this if isinstance(obj, exp.Schema) else obj
        return _full_name(table) if isinstance(table, exp.Table) else None
    return None


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM_PROMPT = """\
You are a data lineage extraction assistant.
Given a SQL script, extract every source→target lineage relationship.
Return ONLY valid JSON — no markdown, no explanation — in this exact shape:
{
  "relationships": [
    {
      "source": {"name": "table_name", "full_path": "catalog.schema.table", "system": "databricks|adls|sql|unknown"},
      "target": {"name": "table_name", "full_path": "catalog.schema.table", "system": "databricks|adls|sql|unknown"},
      "transformation": "brief description of what the SQL does"
    }
  ]
}
If a table is an ADLS/blob path use system "adls". Use "databricks" for 3-part Unity Catalog names.
If you find nothing, return {"relationships": []}."""


def _parse_with_llm(sql: str, filename: str) -> LineageFragment:
    fragment = LineageFragment(artifact_name=filename, artifact_type="sql")
    node_registry: dict[str, LineageNode] = {}

    try:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment,
            api_version="2024-08-01-preview",
        )
        response = llm.invoke([
            SystemMessage(content=_FALLBACK_SYSTEM_PROMPT),
            HumanMessage(content=f"SQL file: {filename}\n\n{sql}"),
        ])
        data = json.loads(response.content)
    except Exception:
        # LLM also failed — return empty fragment so the gap detector catches it
        return fragment

    for rel in data.get("relationships", []):
        src_info = _normalise_llm_node(rel.get("source", {}))
        tgt_info = _normalise_llm_node(rel.get("target", {}))
        if not src_info or not tgt_info:
            continue

        src_node = _get_or_create_node(src_info, node_registry, fragment)
        tgt_node = _get_or_create_node(tgt_info, node_registry, fragment)
        fragment.edges.append(LineageEdge(
            source_id=src_node.id,
            target_id=tgt_node.id,
            transformation=rel.get("transformation"),
            confidence=0.60,
            confidence_level=ConfidenceLevel.MEDIUM,
            evidence=f"LLM fallback from {filename}",
        ))

    return fragment


def _normalise_llm_node(node: dict) -> dict | None:
    name = node.get("name") or node.get("full_path", "").split(".")[-1]
    full_path = node.get("full_path") or name
    system_str = node.get("system", "unknown")
    try:
        system = SystemType(system_str)
    except ValueError:
        system = SystemType.UNKNOWN
    if not name:
        return None
    return {"name": name, "full_path": full_path, "system": system}
