from __future__ import annotations
import json

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.parsers.base_parser import BaseParser
from src.parsers.sql_parser import _get_or_create_node, _normalise_llm_node
from src.graph.models import (
    LineageFragment, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from config.settings import settings

# Namespace prefix for ADF dataset reference names.
# Using this prefix lets the stitcher match pipeline dataset refs (adf://Name)
# to nodes emitted by dataset JSON parsers (also adf://Name), even though
# they come from different uploaded files.
_ADF_NS = "adf://"

# Map ADF dataset type strings → SystemType
_DS_TYPE_SYSTEM: dict[str, SystemType] = {
    # ADLS / Blob file-based
    "AzureBlobFSFile":           SystemType.ADLS,
    "AzureDataLakeStoreFile":    SystemType.ADLS,
    "AzureBlobStorageFile":      SystemType.ADLS,
    "AzureBlobObject":           SystemType.ADLS,
    "Parquet":                   SystemType.ADLS,
    "DelimitedText":             SystemType.ADLS,
    "Json":                      SystemType.ADLS,
    "Orc":                       SystemType.ADLS,
    "Binary":                    SystemType.ADLS,
    "Excel":                     SystemType.ADLS,
    # SQL
    "AzureSqlTable":             SystemType.SQL,
    "SqlServerTable":            SystemType.SQL,
    "AzureSqlDWTable":           SystemType.SQL,
    "AzureSqlMITable":           SystemType.SQL,
    "SnowflakeTable":            SystemType.SQL,
    "SnowflakeV2Table":          SystemType.SQL,
    # Databricks
    "AzureDatabricksDeltaLakeDataset": SystemType.DATABRICKS,
}

# Container activity types that hold nested activities
_CONTAINER_TYPES = {"ForEach", "IfCondition", "Switch", "Until", "ExecutePipeline"}


# ---------------------------------------------------------------------------
# Parser entry point
# ---------------------------------------------------------------------------

class AdfParser(BaseParser):
    """
    Parses Azure Data Factory JSON files.

    Handles three file types emitted by ADF exports:
      - Pipeline JSON  → Copy / DataFlow / Notebook activities → source/sink edges
      - Dataset JSON   → enriches adf://DatasetName nodes with system type and path
      - Linked Service → skipped (no direct lineage, referenced via datasets)

    Pipeline and dataset files are parsed independently; the stitcher connects
    them via shared adf://DatasetName full_path identifiers.
    """

    def can_parse(self, artifact: dict) -> bool:
        filename = artifact.get("filename", "").lower()
        if not filename.endswith(".json"):
            return False
        if artifact.get("artifact_type") == "adf":
            return True
        # Content-sniff: look for ADF-shaped JSON
        content = artifact.get("content", "")
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return False
        props = data.get("properties", {})
        return "activities" in props or "linkedServiceName" in props

    def parse(self, artifact: dict) -> LineageFragment:
        content = artifact.get("content", "")
        filename = artifact.get("filename", "unknown.json")
        fragment = LineageFragment(artifact_name=filename, artifact_type="adf")

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return _parse_with_llm_adf(content, filename)

        props = data.get("properties", {})
        file_type = _detect_file_type(props)

        try:
            if file_type == "pipeline":
                _parse_pipeline(props, filename, fragment)
            elif file_type == "dataset":
                _parse_dataset(data.get("name", ""), props, fragment)
            # linked services: no lineage to extract
        except Exception:
            pass

        if not fragment.edges and file_type == "pipeline":
            return _parse_with_llm_adf(content, filename)

        return fragment


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def _detect_file_type(props: dict) -> str:
    if "activities" in props:
        return "pipeline"
    if "linkedServiceName" in props:
        return "dataset"
    return "linkedservice"


# ---------------------------------------------------------------------------
# Pipeline parsing
# ---------------------------------------------------------------------------

def _parse_pipeline(props: dict, filename: str, fragment: LineageFragment) -> None:
    node_registry: dict[str, LineageNode] = {}
    activities = props.get("activities", [])

    for act in _flatten_activities(activities):
        _process_activity(act, filename, fragment, node_registry)


def _flatten_activities(activities: list[dict]) -> list[dict]:
    """Recursively yield all leaf activities from nested container types."""
    result = []
    for act in activities:
        result.append(act)
        tp = act.get("typeProperties", {})
        act_type = act.get("type", "")

        if act_type == "ForEach":
            result.extend(_flatten_activities(tp.get("activities", [])))
        elif act_type == "IfCondition":
            result.extend(_flatten_activities(tp.get("ifTrueActivities", [])))
            result.extend(_flatten_activities(tp.get("ifFalseActivities", [])))
        elif act_type == "Switch":
            for case in tp.get("cases", []):
                result.extend(_flatten_activities(case.get("activities", [])))
            result.extend(_flatten_activities(tp.get("defaultActivities", [])))
        elif act_type in ("Until", "ExecutePipeline"):
            result.extend(_flatten_activities(tp.get("activities", [])))

    return result


def _process_activity(
    act: dict,
    filename: str,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    act_type = act.get("type", "")
    act_name = act.get("name", act_type)

    if act_type == "Copy":
        _process_copy(act, act_name, filename, fragment, node_registry)
    elif act_type == "ExecuteDataFlow":
        _process_dataflow(act, act_name, filename, fragment, node_registry)
    elif act_type in ("DatabricksNotebook", "DatabricksSparkJar", "DatabricksSparkPython"):
        _process_databricks_activity(act, act_type, act_name, filename, fragment, node_registry)
    elif act_type == "Lookup":
        _process_lookup(act, act_name, filename, fragment, node_registry)


def _process_copy(
    act: dict,
    act_name: str,
    filename: str,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    inputs  = act.get("inputs",  [])
    outputs = act.get("outputs", [])
    tp = act.get("typeProperties", {})
    transform = f"copy: {act_name}"

    # Prefer dataset references; fall back to inline paths
    src_refs = [_ADF_NS + i["referenceName"] for i in inputs  if i.get("referenceName")]
    tgt_refs = [_ADF_NS + o["referenceName"] for o in outputs if o.get("referenceName")]

    # If no dataset refs, try to extract inline paths from typeProperties
    if not src_refs:
        path = _inline_path(tp.get("source", {}))
        if path:
            src_refs = [path]
    if not tgt_refs:
        path = _inline_path(tp.get("sink", {}))
        if path:
            tgt_refs = [path]

    for src_ref in src_refs:
        src_node = _make_ref_node(src_ref, node_registry, fragment)
        for tgt_ref in tgt_refs:
            tgt_node = _make_ref_node(tgt_ref, node_registry, fragment)
            fragment.edges.append(LineageEdge(
                source_id=src_node.id,
                target_id=tgt_node.id,
                transformation=transform,
                confidence=0.85,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=f"parsed from {filename} via ADF pipeline JSON",
            ))


def _process_dataflow(
    act: dict,
    act_name: str,
    filename: str,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    tp = act.get("typeProperties", {})
    df_name = tp.get("dataflow", {}).get("referenceName", act_name)

    inputs  = act.get("inputs",  [])
    outputs = act.get("outputs", [])

    src_refs = [_ADF_NS + i["referenceName"] for i in inputs  if i.get("referenceName")]
    tgt_refs = [_ADF_NS + o["referenceName"] for o in outputs if o.get("referenceName")]

    transform = f"dataflow: {df_name}"
    for src_ref in src_refs:
        src_node = _make_ref_node(src_ref, node_registry, fragment)
        for tgt_ref in tgt_refs:
            tgt_node = _make_ref_node(tgt_ref, node_registry, fragment)
            fragment.edges.append(LineageEdge(
                source_id=src_node.id,
                target_id=tgt_node.id,
                transformation=transform,
                confidence=0.80,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=f"parsed from {filename} via ADF pipeline JSON",
            ))


def _process_databricks_activity(
    act: dict,
    act_type: str,
    act_name: str,
    filename: str,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    tp = act.get("typeProperties", {})
    inputs  = act.get("inputs",  [])
    outputs = act.get("outputs", [])

    if act_type == "DatabricksNotebook":
        nb_path = tp.get("notebookPath", "")
        transform = f"databricks notebook: {nb_path}" if nb_path else f"databricks notebook: {act_name}"
    else:
        transform = f"{act_type.lower()}: {act_name}"

    src_refs = [_ADF_NS + i["referenceName"] for i in inputs  if i.get("referenceName")]
    tgt_refs = [_ADF_NS + o["referenceName"] for o in outputs if o.get("referenceName")]

    for src_ref in src_refs:
        src_node = _make_ref_node(src_ref, node_registry, fragment)
        for tgt_ref in tgt_refs:
            tgt_node = _make_ref_node(tgt_ref, node_registry, fragment)
            fragment.edges.append(LineageEdge(
                source_id=src_node.id,
                target_id=tgt_node.id,
                transformation=transform,
                confidence=0.75,
                confidence_level=ConfidenceLevel.HIGH,
                evidence=f"parsed from {filename} via ADF pipeline JSON",
            ))


def _process_lookup(
    act: dict,
    act_name: str,
    filename: str,
    fragment: LineageFragment,
    node_registry: dict[str, LineageNode],
) -> None:
    # Lookup only reads — add source node with no outgoing edge from parser
    tp = act.get("typeProperties", {})
    ds_ref = tp.get("dataset", {}).get("referenceName")
    if ds_ref:
        _make_ref_node(_ADF_NS + ds_ref, node_registry, fragment)


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------

def _parse_dataset(name: str, props: dict, fragment: LineageFragment) -> None:
    """
    Emit a node for the dataset so the stitcher can connect it to pipeline
    edges that reference it by the adf://Name identifier.
    System type is inferred from the ADF dataset type.
    The actual storage path is recorded in the node description.
    """
    if not name:
        return

    ds_type = props.get("type", "")
    system = _DS_TYPE_SYSTEM.get(ds_type, SystemType.UNKNOWN)
    tp = props.get("typeProperties", {})
    storage_path = _extract_dataset_path(ds_type, tp)

    node = LineageNode(
        name=name,
        system=system,
        full_path=_ADF_NS + name,
        description=storage_path or None,
    )
    fragment.nodes.append(node)


def _extract_dataset_path(ds_type: str, tp: dict) -> str:
    """
    Best-effort extraction of the human-readable storage path
    from a dataset's typeProperties.
    """
    # AzureBlobFS / ADLS Gen2 — new location block
    loc = tp.get("location", {})
    if loc:
        fs   = loc.get("fileSystem", "")
        folder = loc.get("folderPath", "")
        fname  = loc.get("fileName", "")
        parts = [p for p in [fs, folder, fname] if p]
        if parts:
            return "/".join(parts)

    # Old-style ADLS / Blob: folderPath + fileName at top level
    folder = tp.get("folderPath", "")
    fname  = tp.get("fileName", "")
    if folder or fname:
        return "/".join(p for p in [folder, fname] if p)

    # SQL: schema + table or just tableName
    schema = tp.get("schema", "")
    table  = tp.get("table", "") or tp.get("tableName", "")
    if table:
        return f"{schema}.{table}" if schema else table

    # Databricks Delta Lake: database + table
    db    = tp.get("database", "")
    table = tp.get("table", "")
    if db or table:
        return f"{db}.{table}" if db and table else db or table

    return ""


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _make_ref_node(
    full_path: str,
    node_registry: dict[str, LineageNode],
    fragment: LineageFragment,
) -> LineageNode:
    """Get or create a node for an adf:// dataset reference or inline path."""
    if full_path in node_registry:
        return node_registry[full_path]

    name = full_path.replace(_ADF_NS, "").split("/")[-1].split(".")[-1]
    system = SystemType.ADLS if full_path.startswith(("abfss://", "wasbs://")) else SystemType.UNKNOWN
    info = {"name": name, "full_path": full_path, "system": system}
    return _get_or_create_node(info, node_registry, fragment)


def _inline_path(side: dict) -> str | None:
    """Extract a literal path from a Copy source/sink typeProperties block."""
    for key in ("fileName", "folderPath", "fileSystem", "tableName", "table"):
        val = side.get(key)
        if val and isinstance(val, str):
            return val
    return None


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_ADF_FALLBACK_PROMPT = """\
You are a data lineage extraction assistant specialising in Azure Data Factory.
Given ADF pipeline JSON, extract every source→target lineage relationship.
Return ONLY valid JSON — no markdown, no explanation:
{
  "relationships": [
    {
      "source": {"name": "DatasetOrPath", "full_path": "adf://DatasetName_or_actual_path", "system": "adls|databricks|sql|unknown"},
      "target": {"name": "DatasetOrPath", "full_path": "adf://DatasetName_or_actual_path", "system": "adls|databricks|sql|unknown"},
      "transformation": "copy | dataflow: name | databricks notebook: /path | etc."
    }
  ]
}
Prefix ADF dataset reference names with adf://.
Use system "adls" for blob/ADLS paths, "databricks" for Delta Lake tables, "sql" for SQL tables.
If you find nothing, return {"relationships": []}."""


def _parse_with_llm_adf(content: str, filename: str) -> LineageFragment:
    fragment = LineageFragment(artifact_name=filename, artifact_type="adf")
    node_registry: dict[str, LineageNode] = {}

    try:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment,
            api_version="2024-08-01-preview",
        )
        response = llm.invoke([
            SystemMessage(content=_ADF_FALLBACK_PROMPT),
            HumanMessage(content=f"ADF pipeline: {filename}\n\n{content}"),
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
