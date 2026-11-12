"""
Unity Catalog metadata export parser.

A UC export carries no transformation logic, so it produces nodes only. The
stitcher uses them as the inventory of concrete tables that exist:

* templated names in notebooks (``action_prod.bronze.pos_{country}_transactions``)
  are pattern-matched against these tables;
* owner / comment metadata enrich nodes discovered by other parsers;
* tables that end up with no lineage edges are dropped from the graph rather
  than shown as noise (the fragment is flagged ``catalog_only``).
"""
from __future__ import annotations

import json

from src.parsers.base_parser import BaseParser
from src.graph.models import LineageFragment, LineageNode, SystemType
from src.graph.stewardship import declared_external_from_comment, producer_from_text


class UcCatalogParser(BaseParser):

    def can_parse(self, artifact: dict) -> bool:
        if artifact.get("artifact_type") == "uc_catalog":
            return True
        if not artifact.get("filename", "").lower().endswith(".json"):
            return False
        data = _load(artifact.get("content", ""))
        return _is_uc_export(data)

    def parse(self, artifact: dict) -> LineageFragment:
        filename = artifact.get("filename", "uc_export.json")
        fragment = LineageFragment(artifact_name=filename, artifact_type="uc_catalog")
        fragment.raw_metadata["catalog_only"] = True
        data = _load(artifact.get("content", "")) or {}

        last_writes: dict[str, dict] = {}
        declared: dict[str, str] = {}
        for t in data.get("tables", []):
            full_name = t.get("full_name")
            if not full_name:
                continue
            producer = (t.get("table_properties") or {}).get("declared_producer") \
                or declared_external_from_comment(t.get("comment")) \
                or _producer_written_outside_databricks(t)
            if producer:
                declared[full_name] = producer
            fragment.nodes.append(LineageNode(
                name=t.get("name") or full_name.split(".")[-1],
                system=SystemType.DATABRICKS,
                full_path=full_name,
                description=t.get("comment") or None,
                owner=t.get("owner") or None,
            ))
            if t.get("last_write"):
                last_writes[full_name] = t["last_write"]

        fragment.raw_metadata["last_write"] = last_writes
        # Applied by the stitcher only to tables that end up in the graph, so a
        # declaration alone does not pull an unused catalog table into view.
        fragment.raw_metadata["declared_producers"] = declared
        fragment.raw_metadata["workspace_id"] = (data.get("export") or {}).get("workspace_id")
        return fragment


def _producer_written_outside_databricks(table: dict) -> str | None:
    """
    The comment names who loads the table ("Maintained by Group Tax; quarterly
    CSV load") AND Unity Catalog recorded a last write made by no Databricks job
    or notebook — the data arrives from outside the platform.

    Tables last written by a job (e.g. store_id_xref, job 612377) are not
    declared external even when the comment names a team: their producer is a
    platform artifact that simply hasn't been uploaded yet.
    """
    lw = table.get("last_write")
    if not isinstance(lw, dict) or lw.get("job_id") or lw.get("notebook_path"):
        return None
    return producer_from_text(table.get("comment"))


def _load(content: str):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def _is_uc_export(data) -> bool:
    if not isinstance(data, dict):
        return False
    tables = data.get("tables")
    if not isinstance(tables, list) or not tables:
        return False
    tool = str((data.get("export") or {}).get("tool", "")).lower()
    return "unity-catalog" in tool or all(isinstance(t, dict) and "full_name" in t for t in tables)
