from __future__ import annotations
import ast
import json
import re

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.parsers.base_parser import BaseParser
from src.parsers.sql_parser import (
    _parse_with_sqlglot,
    _get_or_create_node,
    _infer_system,
    _normalise_llm_node,
)
from src.parsers.notebook_parser import _extract_read_paths
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from config.settings import settings

# Marker schema injected during SQL DLT preprocessing so sqlglot can parse
# LIVE.table references — detected post-parse to assign SystemType.DLT
_DLT_SCHEMA_MARKER = "_dlt_"

# Operation attributes used to summarise transformations
_TRANSFORM_ATTRS = {
    "filter": "filtered", "where": "filtered",
    "join": "joined", "crossJoin": "joined",
    "groupBy": "aggregated", "agg": "aggregated",
    "withColumn": "columns computed", "withColumnRenamed": "columns renamed",
    "union": "unioned", "unionAll": "unioned", "unionByName": "unioned",
    "distinct": "deduplicated",
    "sort": "sorted", "orderBy": "sorted",
}


# ---------------------------------------------------------------------------
# Parser entry point
# ---------------------------------------------------------------------------

class DltParser(BaseParser):
    """
    Parses Delta Live Tables pipeline definitions.
    Python (.py): AST-based — finds @dlt.table / @dlt.view decorated functions
                  and their dlt.read() / spark.read... source calls.
    SQL (.sql):   preprocesses LIVE TABLE / LIVE. syntax, then sqlglot.
    Falls back to LLM when neither path yields edges.
    """

    def can_parse(self, artifact: dict) -> bool:
        filename = artifact.get("filename", "").lower()
        content = artifact.get("content", "")
        if artifact.get("artifact_type") == "dlt":
            return True
        if filename.endswith(".ipynb"):
            return False   # notebooks are handled by NotebookParser
        if filename.endswith(".py") and ("@dlt." in content or "import dlt" in content):
            return True
        if filename.endswith(".sql") and re.search(r"\bLIVE\b", content, re.IGNORECASE):
            return True
        return False

    def parse(self, artifact: dict) -> LineageFragment:
        content = artifact.get("content", "")
        filename = artifact.get("filename", "unknown")

        is_sql = filename.lower().endswith(".sql") or _looks_like_sql_dlt(content)

        try:
            fragment = _parse_sql_dlt(content, filename) if is_sql else _parse_python_dlt(content, filename)
            if fragment.edges:
                return fragment
        except Exception:
            pass

        return _parse_with_llm_dlt(content, filename)


# ---------------------------------------------------------------------------
# Python DLT path — AST-based
# ---------------------------------------------------------------------------

def _parse_python_dlt(content: str, filename: str) -> LineageFragment:
    tree = ast.parse(content)
    source_lines = content.splitlines()
    visitor = _DltVisitor(source_lines)
    visitor.visit(tree)

    fragment = LineageFragment(artifact_name=filename, artifact_type="dlt")
    node_registry: dict[str, LineageNode] = {}

    for defn in visitor.definitions:
        tgt_info = {"name": defn["target"], "full_path": defn["target"], "system": SystemType.DLT}
        tgt_node = _get_or_create_node(tgt_info, node_registry, fragment)
        transform = _summarise_ops(defn["ops"]) if defn["ops"] else None

        for src_path, src_type in defn["sources"]:
            system = SystemType.DLT if src_type == "dlt" else _infer_system(src_path, _catalog(src_path))
            src_info = {"name": src_path.split(".")[-1], "full_path": src_path, "system": system}
            src_node = _get_or_create_node(src_info, node_registry, fragment)
            fragment.edges.append(LineageEdge(
                source_id=src_node.id,
                target_id=tgt_node.id,
                transformation=transform,
                confidence=0.90,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=f"parsed from {filename} via ast",
            ))

    return fragment


class _DltVisitor(ast.NodeVisitor):
    """Walks a Python DLT pipeline AST and collects table definitions."""

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.definitions: list[dict] = []  # [{target, sources, ops}]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- internals --

    def _process_function(self, node: ast.FunctionDef) -> None:
        decorator = self._find_dlt_decorator(node)
        if not decorator:
            return

        target = self._extract_target_name(decorator, node.name)
        sources = self._collect_sources(node)
        ops = self._collect_ops(node)

        if sources:
            self.definitions.append({"target": target, "sources": sources, "ops": ops})

    def _find_dlt_decorator(self, node: ast.FunctionDef) -> ast.expr | None:
        for d in node.decorator_list:
            if _is_dlt_decorator(d):
                return d
        return None

    def _extract_target_name(self, decorator: ast.expr, func_name: str) -> str:
        if isinstance(decorator, ast.Call):
            for kw in decorator.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
        return func_name

    def _collect_sources(self, node: ast.FunctionDef) -> list[tuple[str, str]]:
        """Return list of (path, kind) where kind is 'dlt' or 'external'."""
        sources: list[tuple[str, str]] = []
        seen: set[str] = set()

        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            path = _extract_dlt_read_path(child)
            if path and path not in seen:
                seen.add(path)
                sources.append((path, "dlt"))

        # Also run notebook regex patterns on the function body source
        body_source = self._body_source(node)
        for path in _extract_read_paths(body_source):
            if path not in seen:
                seen.add(path)
                sources.append((path, "external"))

        return sources

    def _collect_ops(self, node: ast.FunctionDef) -> list[str]:
        ops: list[str] = []
        seen: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in _TRANSFORM_ATTRS:
                label = _TRANSFORM_ATTRS[child.attr]
                if label not in seen:
                    seen.add(label)
                    ops.append(label)
        return ops

    def _body_source(self, node: ast.FunctionDef) -> str:
        if not node.body:
            return ""
        start = node.body[0].lineno - 1
        end = getattr(node, "end_lineno", start + 100)
        return "\n".join(self.source_lines[start:end])


def _is_dlt_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "dlt"
            and node.attr in ("table", "view", "table_aggregate")
        )
    if isinstance(node, ast.Call):
        return _is_dlt_decorator(node.func)
    return False


def _extract_dlt_read_path(call: ast.Call) -> str | None:
    """Return the table name from dlt.read("x") or dlt.read_stream("x"), else None."""
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "dlt"
        and func.attr in ("read", "read_stream")
        and call.args
        and isinstance(call.args[0], ast.Constant)
    ):
        return str(call.args[0].value)
    return None


# ---------------------------------------------------------------------------
# SQL DLT path — preprocess then sqlglot
# ---------------------------------------------------------------------------

# Regex for DLT-specific SQL keywords
_RE_CREATE_LIVE = re.compile(
    r"CREATE\s+(?:OR\s+REFRESH\s+)?(?:STREAMING\s+)?LIVE\s+(TABLE|VIEW)",
    re.IGNORECASE,
)
_RE_STREAM_LIVE = re.compile(r"STREAM\s*\(\s*LIVE\.(\w+)\s*\)", re.IGNORECASE)
_RE_LIVE_REF    = re.compile(r"\bLIVE\.(\w+)\b", re.IGNORECASE)


def _preprocess_dlt_sql(sql: str) -> str:
    """
    Strip DLT-specific syntax so sqlglot can parse the SQL normally.
    LIVE.table references are renamed to _dlt_.table so we can identify
    them as DLT nodes after parsing.
    """
    sql = _RE_CREATE_LIVE.sub(r"CREATE \1", sql)
    sql = _RE_STREAM_LIVE.sub(rf"{_DLT_SCHEMA_MARKER}.\1", sql)
    sql = _RE_LIVE_REF.sub(rf"{_DLT_SCHEMA_MARKER}.\1", sql)
    return sql


def _parse_sql_dlt(content: str, filename: str) -> LineageFragment:
    processed = _preprocess_dlt_sql(content)
    fragment = _parse_with_sqlglot(processed, filename)

    # Override artifact_type and fix system types for DLT-marked nodes
    fragment.artifact_type = "dlt"
    for node in fragment.nodes:
        if node.full_path.startswith(f"{_DLT_SCHEMA_MARKER}."):
            node.full_path = node.full_path[len(_DLT_SCHEMA_MARKER) + 1:]
            node.name = node.full_path
            node.system = SystemType.DLT

    # Update confidence evidence
    for edge in fragment.edges:
        edge.evidence = f"parsed from {filename} via sqlglot (DLT SQL)"

    return fragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_sql_dlt(content: str) -> bool:
    return bool(re.search(r"\bLIVE\s+(?:TABLE|VIEW)\b", content, re.IGNORECASE))


def _catalog(path: str) -> str | None:
    parts = path.split(".")
    return parts[0] if len(parts) >= 3 else None


def _summarise_ops(ops: list[str]) -> str:
    labels = list(dict.fromkeys(ops))
    return ", ".join(labels) if labels else "dlt transformation"


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_DLT_FALLBACK_PROMPT = """\
You are a data lineage extraction assistant specialising in Databricks Delta Live Tables.
Given DLT pipeline code (Python or SQL), extract every source→target lineage relationship.
Return ONLY valid JSON — no markdown, no explanation:
{
  "relationships": [
    {
      "source": {"name": "table_name", "full_path": "table_name_or_catalog.schema.table", "system": "dlt|databricks|adls|sql|unknown"},
      "target": {"name": "table_name", "full_path": "table_name", "system": "dlt"},
      "transformation": "brief description or null"
    }
  ]
}
Use system "dlt" for tables defined within the DLT pipeline (dlt.read() sources and decorated function targets).
Use "adls" for abfss://, wasbs://, dbfs:/mnt/ paths.
Use "databricks" for 3-part Unity Catalog names (catalog.schema.table).
If you find nothing, return {"relationships": []}."""


def _parse_with_llm_dlt(content: str, filename: str) -> LineageFragment:
    fragment = LineageFragment(artifact_name=filename, artifact_type="dlt")
    node_registry: dict[str, LineageNode] = {}

    try:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment,
            api_version="2024-08-01-preview",
        )
        response = llm.invoke([
            SystemMessage(content=_DLT_FALLBACK_PROMPT),
            HumanMessage(content=f"DLT pipeline: {filename}\n\n{content}"),
        ])
        data = json.loads(response.content)
    except Exception:
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
