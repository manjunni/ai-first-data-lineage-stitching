from __future__ import annotations
import json
import re
import uuid
from dataclasses import dataclass, field

import nbformat

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.parsers.base_parser import BaseParser
from src.parsers.sql_parser import (
    _parse_with_sqlglot,
    _get_or_create_node,
    _infer_system,
    _normalise_llm_node,
)
from src.graph.models import (
    ColumnMapping, LineageFragment, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from config.settings import settings

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches either "..." or '...' and captures the inner string
_Q = r"""["']([^"']+)["']"""

# -- Reads --
_READ_TABLE    = re.compile(r'spark\.read\b.*?\.table\s*\(\s*' + _Q + r'\s*\)',     re.DOTALL)
_READ_LOAD     = re.compile(r'spark\.read\b.*?\.(?:load|parquet|csv|json|orc|text)\s*\(\s*' + _Q + r'\s*\)', re.DOTALL)
_SPARK_TABLE   = re.compile(r'spark\.table\s*\(\s*' + _Q + r'\s*\)')
_DELTA_NAME    = re.compile(r'DeltaTable\.forName\s*\(\s*(?:spark\s*,\s*)?' + _Q + r'\s*\)')
_DELTA_PATH    = re.compile(r'DeltaTable\.forPath\s*\(\s*(?:spark\s*,\s*)?' + _Q + r'\s*\)')

# -- Writes --
_WRITE_TABLE   = re.compile(r'\.write\b.*?\.saveAsTable\s*\(\s*' + _Q + r'\s*\)',  re.DOTALL)
_WRITE_INSERT  = re.compile(r'\.write\b.*?\.insertInto\s*\(\s*' + _Q + r'\s*\)',   re.DOTALL)
_WRITE_SAVE    = re.compile(r'\.write\b.*?\.(?:save|parquet|csv|json|orc|text)\s*\(\s*' + _Q + r'\s*\)', re.DOTALL)
_WRITE_TO      = re.compile(r'\.writeTo\s*\(\s*' + _Q + r'\s*\)')

# -- Variable tracking --
# var = spark.read... / spark.table(...) / DeltaTable.for...
_ASSIGN_READ   = re.compile(r'^(\w+)\s*=\s*(?:spark\.(?:read|table)\b|DeltaTable\.for)', re.MULTILINE)
# var2 = var1.transform(...)
_ASSIGN_DERIVE = re.compile(
    r'^(\w+)\s*=\s*(\w+)\.'
    r'(?:filter|where|select|withColumn|withColumnRenamed|join|crossJoin'
    r'|groupBy|agg|alias|drop|dropna|fillna|na\.\w+'
    r'|limit|distinct|union|unionAll|unionByName'
    r'|cache|persist|repartition|coalesce|sort|orderBy|sample)',
    re.MULTILINE,
)
# Detect that a write is present in a logical line
_HAS_WRITE     = re.compile(r'\.write(?:To)?\b')
# The leading variable in a statement — first identifier, ignoring a leading '('
_LEADING_VAR   = re.compile(r'(?:^|\()\s*(\w+)\b')
# operation names for transformation summary
_DERIVE_OP     = re.compile(
    r'\.(?P<op>filter|where|join|crossJoin|groupBy|agg|withColumn|withColumnRenamed'
    r'|union|unionAll|unionByName|distinct|sort|orderBy)\b'
)

# spark.sql("...") — handles triple- and single-quoted strings
_SPARK_SQL = re.compile(
    r'spark\.sql\s*\(\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\'|"(.*?)"|\'(.*?)\')\s*\)',
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# DataFlowTracker — simple cross-cell variable lineage
# ---------------------------------------------------------------------------

@dataclass
class DataFlowTracker:
    reads:   dict[str, list[str]]  = field(default_factory=dict)  # var → [source paths]
    derives: dict[str, str]        = field(default_factory=dict)  # child → parent var
    ops:     dict[str, list[str]]  = field(default_factory=dict)  # var → [operations applied]
    mappings: dict[str, list[ColumnMapping]] = field(default_factory=dict)

    def record_read(self, var: str, paths: list[str]) -> None:
        self.reads[var] = paths

    def record_derive(self, child: str, parent: str, operations: list[str], line: str = "") -> None:
        if child == parent:
            self.ops.setdefault(child, []).extend(operations)
            self.mappings.setdefault(child, []).extend(_extract_dataframe_mappings(line))
            return
        self.derives[child] = parent
        self.ops[child] = operations
        self.mappings[child] = _extract_dataframe_mappings(line)

    def get_sources(self, var: str, _seen: frozenset[str] = frozenset()) -> list[str]:
        if var in _seen:
            return []
        if var in self.reads:
            return self.reads[var]
        parent = self.derives.get(var)
        if parent:
            return self.get_sources(parent, _seen | {var})
        return []

    def get_operations(self, var: str, _seen: frozenset[str] = frozenset()) -> list[str]:
        """Collect all transformation operations between source and write."""
        if var in _seen:
            return []
        ops = list(self.ops.get(var, []))
        parent = self.derives.get(var)
        if parent:
            ops = self.get_operations(parent, _seen | {var}) + ops
        return ops

    def get_mappings(self, var: str, _seen: frozenset[str] = frozenset()) -> list[ColumnMapping]:
        if var in _seen:
            return []
        parent = self.derives.get(var)
        inherited = self.get_mappings(parent, _seen | {var}) if parent else []
        return inherited + list(self.mappings.get(var, []))


# ---------------------------------------------------------------------------
# Public parser class
# ---------------------------------------------------------------------------

class NotebookParser(BaseParser):
    """
    Parses Databricks .ipynb notebooks to extract data lineage.
    Primary path: nbformat + regex over Spark DataFrame API calls.
    Handles %sql magic cells and spark.sql("...") via the SQL parser (sqlglot).
    Falls back to LLM when regex yields no edges.
    """

    def can_parse(self, artifact: dict) -> bool:
        filename = artifact.get("filename", "").lower()
        if filename.endswith(".ipynb"):
            return True
        if artifact.get("artifact_type") == "notebook":
            return True
        # Non-DLT Databricks Python scripts with Spark patterns
        if filename.endswith(".py"):
            content = artifact.get("content", "")
            has_spark = "spark." in content or "SparkSession" in content or "dbutils." in content
            is_dlt = "@dlt." in content or "import dlt" in content
            return has_spark and not is_dlt
        return False

    def parse(self, artifact: dict) -> LineageFragment:
        content = artifact.get("content", "")
        filename = artifact.get("filename", "unknown")

        if not filename.lower().endswith(".ipynb"):
            # Plain Python Databricks file — apply regex directly without nbformat
            try:
                fragment = _parse_python_file(content, filename)
                if fragment.edges:
                    return fragment
            except Exception:
                pass
            return _parse_with_llm_nb(content, filename)

        try:
            fragment = _parse_with_nbformat(content, filename)
            if fragment.edges:
                return fragment
        except Exception:
            pass

        return _parse_with_llm_nb(content, filename)


# ---------------------------------------------------------------------------
# nbformat path
# ---------------------------------------------------------------------------

def _parse_with_nbformat(content: str, filename: str) -> LineageFragment:
    nb = nbformat.reads(content, as_version=4)
    fragment = LineageFragment(artifact_name=filename, artifact_type="notebook")
    node_registry: dict[str, LineageNode] = {}
    tracker = DataFlowTracker()

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source.strip()
        if not source:
            continue

        logical_lines = _join_logical_lines(source)

        # %sql magic cell — delegate entirely to sqlglot
        if source.startswith("%sql"):
            sql_text = source[4:].strip()
            _absorb_sql(sql_text, filename, fragment, node_registry)
            continue

        for line in logical_lines:
            # spark.sql("...") calls — extract SQL and delegate
            for sql_text in _extract_spark_sql(line):
                _absorb_sql(sql_text, filename, fragment, node_registry)

            # Variable reads: var = spark.read.table(...) etc.
            _process_read_assignment(line, tracker)

            # Variable transforms: var2 = var1.filter(...)
            _process_derive_assignment(line, tracker)

    # Walk all logical lines again to find writes and create edges
    all_code = "\n".join(
        cell.source
        for cell in nb.cells
        if cell.cell_type == "code" and not cell.source.strip().startswith("%sql")
    )
    for line in _join_logical_lines(all_code):
        _process_write(line, filename, tracker, fragment, node_registry)

    return fragment


def _process_read_assignment(line: str, tracker: DataFlowTracker) -> None:
    """
    If a line assigns a variable from spark.read/spark.table/DeltaTable,
    record the source paths for that variable.
    """
    m = _ASSIGN_READ.search(line)
    if not m:
        return
    var = m.group(1)
    paths = _extract_read_paths(line)
    if paths:
        tracker.record_read(var, paths)


def _process_derive_assignment(line: str, tracker: DataFlowTracker) -> None:
    """If var2 = var1.op(...), record the parent→child relationship."""
    m = _ASSIGN_DERIVE.search(line)
    if not m:
        return
    child, parent = m.group(1), m.group(2)
    ops = [m2.group("op") for m2 in _DERIVE_OP.finditer(line)]
    tracker.record_derive(child, parent, ops, line)


def _process_write(
    line: str,
    filename: str,
    tracker: DataFlowTracker,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    """
    If a line contains a write, resolve it back to its source reads
    and create LineageEdge entries.
    Handles: var.write..., var.method().write..., and (var .method() .write...)
    """
    if not _HAS_WRITE.search(line):
        return

    write_var_m = _LEADING_VAR.search(line)
    if not write_var_m:
        return

    write_var = write_var_m.group(1)
    # Skip Python keywords that can appear at line start
    if write_var in ("if", "for", "while", "with", "return", "from", "import", "class", "def", "not", "and", "or"):
        return
    target_paths = _extract_write_paths(line)
    source_paths = tracker.get_sources(write_var)
    ops = tracker.get_operations(write_var)
    mappings = tracker.get_mappings(write_var)
    transform = _summarise_ops(ops) if ops else "spark transformation"

    for tgt_path in target_paths:
        tgt_info = _path_to_info(tgt_path)
        tgt_node = _get_or_create_node(tgt_info, node_registry, fragment)

        for src_path in source_paths:
            src_info = _path_to_info(src_path)
            src_node = _get_or_create_node(src_info, node_registry, fragment)
            fragment.edges.append(LineageEdge(
                source_id=src_node.id,
                target_id=tgt_node.id,
                transformation=transform,
                confidence=0.85,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=f"parsed from {filename} via regex",
                column_mappings=mappings,
            ))


def _extract_dataframe_mappings(line: str) -> list[ColumnMapping]:
    """Extract mappings from common PySpark select/withColumn expressions."""
    mappings: list[ColumnMapping] = []
    evidence = "parsed from PySpark DataFrame API"

    for match in re.finditer(r'\.withColumn\s*\(\s*["\']([^"\']+)["\']\s*,\s*(.*)\)\s*$', line):
        target, expression = match.groups()
        sources = re.findall(r'(?:col|column)\s*\(\s*["\']([^"\']+)["\']', expression)
        sources = sources or re.findall(r'\b([A-Za-z_]\w*)\b', expression)
        sources = [s for s in dict.fromkeys(sources) if s not in {"lit", "when", "otherwise"}]
        for source in sources:
            mappings.append(ColumnMapping(
                source_column=source,
                target_column=target,
                transformation=expression.strip(),
                confidence=0.85,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=evidence,
            ))

    select_match = re.search(r'\.select\s*\((.*)\)', line)
    if select_match:
        for item in re.split(r',\s*(?![^()]*\))', select_match.group(1)):
            item = item.strip()
            alias = re.search(r'\.alias\s*\(\s*["\']([^"\']+)["\']', item)
            target = alias.group(1) if alias else None
            source_match = re.search(r'(?:col\s*\(\s*)?["\']?([A-Za-z_]\w*)["\']?\s*\)?', item)
            if not source_match:
                continue
            source = source_match.group(1)
            target = target or source
            transformation = item if alias else None
            mappings.append(ColumnMapping(
                source_column=source,
                target_column=target,
                transformation=transformation,
                confidence=0.85,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=evidence,
            ))
    return mappings


def _absorb_sql(
    sql_text: str,
    filename: str,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    """Parse a SQL string with sqlglot and merge its nodes/edges into fragment."""
    try:
        sql_frag = _parse_with_sqlglot(sql_text, filename)
    except Exception:
        return

    # Tables referenced inside spark.sql() are Hive/Unity Catalog (Databricks),
    # not SQL Server — override the SQL default from _infer_system.
    # Build an id-remap so that edges always reference the canonical node id
    # (a node with the same full_path may have been registered by an earlier
    # spark.sql() call with a different uuid).
    id_remap: dict[str, str] = {}
    for node in sql_frag.nodes:
        if node.system == SystemType.SQL:
            node.system = SystemType.DATABRICKS
        # Skip ghost nodes produced by CREATE SCHEMA (empty name / trailing dot)
        if not node.name or node.full_path.endswith("."):
            continue
        if node.full_path in node_registry:
            id_remap[node.id] = node_registry[node.full_path].id
        else:
            node_registry[node.full_path] = node
            fragment.nodes.append(node)
            id_remap[node.id] = node.id

    for edge in sql_frag.edges:
        src_id = id_remap.get(edge.source_id, edge.source_id)
        tgt_id = id_remap.get(edge.target_id, edge.target_id)
        fragment.edges.append(edge.model_copy(update={"source_id": src_id, "target_id": tgt_id}))


# ---------------------------------------------------------------------------
# Path extraction helpers
# ---------------------------------------------------------------------------

def _extract_read_paths(line: str) -> list[str]:
    paths: list[str] = []
    for pattern in (_READ_TABLE, _READ_LOAD, _SPARK_TABLE, _DELTA_NAME, _DELTA_PATH):
        for m in pattern.finditer(line):
            paths.append(m.group(1))
    return list(dict.fromkeys(paths))  # deduplicate, preserve order


def _extract_write_paths(line: str) -> list[str]:
    paths: list[str] = []
    for pattern in (_WRITE_TABLE, _WRITE_INSERT, _WRITE_SAVE, _WRITE_TO):
        for m in pattern.finditer(line):
            paths.append(m.group(1))
    return list(dict.fromkeys(paths))


def _extract_spark_sql(line: str) -> list[str]:
    results = []
    for m in _SPARK_SQL.finditer(line):
        # Groups 1-4 correspond to the four quote alternatives
        sql_text = next((g for g in m.groups() if g is not None), None)
        if sql_text:
            results.append(sql_text.strip())
    return results


def _path_to_info(path: str) -> dict:
    parts = path.split(".")
    name = parts[-1]
    catalog = parts[0] if len(parts) >= 3 else None
    system = _infer_system(path, catalog)
    # In a Spark notebook, simple/2-part table names are Hive Metastore /
    # Unity Catalog tables (Databricks), not SQL Server tables.
    if system == SystemType.SQL:
        system = SystemType.DATABRICKS
    return {
        "name": name,
        "full_path": path,
        "system": system,
    }


def _summarise_ops(ops: list[str]) -> str:
    _LABELS = {
        "filter": "filtered", "where": "filtered",
        "join": "joined", "crossJoin": "joined",
        "groupBy": "aggregated", "agg": "aggregated",
        "withColumn": "columns computed", "withColumnRenamed": "columns renamed",
        "union": "unioned", "unionAll": "unioned", "unionByName": "unioned",
        "distinct": "deduplicated",
        "sort": "sorted", "orderBy": "sorted",
    }
    labels = list(dict.fromkeys(_LABELS.get(op, op) for op in ops))
    return ", ".join(labels) if labels else "spark transformation"


# ---------------------------------------------------------------------------
# Logical line joining — handles backslash and open-paren continuations
# ---------------------------------------------------------------------------

def _join_logical_lines(source: str) -> list[str]:
    """
    Join continuation lines so multi-line spark chains become single strings
    that the regexes can match end-to-end.
    """
    raw = source.splitlines()
    logical: list[str] = []
    buf = ""
    depth = 0

    for raw_line in raw:
        stripped = raw_line.rstrip()
        depth += stripped.count("(") - stripped.count(")")
        depth = max(depth, 0)

        if stripped.endswith("\\"):
            buf += stripped[:-1].rstrip() + " "
        elif depth > 0:
            buf += stripped + " "
        else:
            combined = (buf + stripped).strip()
            buf = ""
            if combined:
                logical.append(combined)

    if buf.strip():
        logical.append(buf.strip())

    return logical


# ---------------------------------------------------------------------------
# Plain Python file parser (non-.ipynb Databricks scripts)
# ---------------------------------------------------------------------------

def _parse_python_file(content: str, filename: str) -> LineageFragment:
    """
    Parse a .py Databricks script using the same regex/tracker logic as
    _parse_with_nbformat but treating the entire file as one code block.
    """
    fragment = LineageFragment(artifact_name=filename, artifact_type="notebook")
    node_registry: dict[str, LineageNode] = {}
    tracker = DataFlowTracker()

    logical_lines = _join_logical_lines(content)

    for line in logical_lines:
        for sql_text in _extract_spark_sql(line):
            _absorb_sql(sql_text, filename, fragment, node_registry)
        _process_read_assignment(line, tracker)
        _process_derive_assignment(line, tracker)

    for line in logical_lines:
        _process_write(line, filename, tracker, fragment, node_registry)

    return fragment


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_NB_FALLBACK_PROMPT = """\
You are a data lineage extraction assistant.
Given the source code of a Databricks notebook, extract every source→target lineage relationship.
Return ONLY valid JSON — no markdown, no explanation:
{
  "relationships": [
    {
      "source": {"name": "table_or_path", "full_path": "catalog.schema.table_or_path", "system": "databricks|adls|sql|unknown"},
      "target": {"name": "table_or_path", "full_path": "catalog.schema.table_or_path", "system": "databricks|adls|sql|unknown"},
      "transformation": "brief description"
    }
  ]
}
Use system "adls" for abfss://, wasbs://, or dbfs:/mnt/... paths.
Use "databricks" for Unity Catalog (3-part) or Hive Metastore table names.
For DataFrame select/withColumn operations, include optional "column_mappings":
[{"source_column": "...", "target_column": "...", "transformation": "..."}].
If you find nothing, return {"relationships": []}."""


def _parse_with_llm_nb(content: str, filename: str) -> LineageFragment:
    fragment = LineageFragment(artifact_name=filename, artifact_type="notebook")
    node_registry: dict[str, LineageNode] = {}

    # Extract raw notebook text for the LLM (concatenate code cells only)
    try:
        nb = nbformat.reads(content, as_version=4)
        code_text = "\n\n".join(
            cell.source for cell in nb.cells if cell.cell_type == "code"
        )
    except Exception:
        code_text = content

    try:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment,
            api_version="2024-08-01-preview",
        )
        response = llm.invoke([
            SystemMessage(content=_NB_FALLBACK_PROMPT),
            HumanMessage(content=f"Notebook: {filename}\n\n{code_text}"),
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
            column_mappings=[
                ColumnMapping(
                    source_column=m["source_column"],
                    target_column=m["target_column"],
                    transformation=m.get("transformation"),
                    confidence=0.60,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    evidence=f"LLM fallback from {filename}",
                )
                for m in rel.get("column_mappings", [])
                if m.get("source_column") and m.get("target_column")
            ],
        ))

    return fragment
