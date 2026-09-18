from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional
import json

from src.graph.models import (
    ColumnMapping,
    LineageGraph,
    LineageNode,
    LineageEdge,
    SystemType,
)
from config.settings import settings

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PublishResult:
    status: str                          # "published" | "dry_run" | "error" | "skipped"
    catalog: str
    run_id: str
    nodes_published: int = 0
    edges_published: int = 0
    errors: list[str]     = field(default_factory=list)
    catalog_urls: list[str] = field(default_factory=list)
    payload: Optional[dict] = None       # serialized payload (audit trail)

    @property
    def success(self) -> bool:
        return self.status in ("published", "dry_run") and not self.errors


# ---------------------------------------------------------------------------
# Alation — system-type → object-type mapping
# ---------------------------------------------------------------------------

_ALATION_OBJECT_TYPE: dict[SystemType, str] = {
    SystemType.ADLS:        "SCHEMA",
    SystemType.DATABRICKS:  "TABLE",
    SystemType.SQL:         "TABLE",
    SystemType.DLT:         "TABLE",
    SystemType.ADF:         "DATAFLOW",
    SystemType.BI:          "REPORT",
    SystemType.UNKNOWN:     "DATAFLOW",
}

# ---------------------------------------------------------------------------
# Collibra — system-type → asset-type display name (actual type UUIDs are
# configured per Collibra instance; we use these names for dry-run clarity)
# ---------------------------------------------------------------------------

_COLLIBRA_ASSET_TYPE_NAME: dict[SystemType, str] = {
    SystemType.ADLS:        "File",
    SystemType.DATABRICKS:  "Table",
    SystemType.SQL:         "Table",
    SystemType.DLT:         "Table",
    SystemType.ADF:         "System",
    SystemType.BI:          "Report",
    SystemType.UNKNOWN:     "Data Asset",
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CatalogPublisher:
    """Base interface for publishing lineage to a catalog tool."""

    catalog_name: str = "unknown"

    def publish(self, graph: LineageGraph, run_id: str) -> PublishResult:
        raise NotImplementedError

    def validate_config(self) -> list[str]:
        """Return a list of missing configuration keys, empty if ready."""
        return []


# ---------------------------------------------------------------------------
# Alation publisher
# ---------------------------------------------------------------------------

class AlationPublisher(CatalogPublisher):
    """
    Publish lineage to Alation via the Integration API v1.

    Endpoint: POST /integration/v1/lineage/
    Auth:     Authorization: TOKEN {api_token}

    One lineage payload is created per edge. Each payload defines a
    named dataflow with its source and target objects.
    When credentials are absent the result is "skipped" with the full
    payload included so it can be reviewed before the real API call.
    """

    catalog_name = "alation"

    def validate_config(self) -> list[str]:
        missing = []
        if not settings.alation_base_url:
            missing.append("ALATION_BASE_URL")
        if not settings.alation_api_token:
            missing.append("ALATION_API_TOKEN")
        return missing

    def publish(self, graph: LineageGraph, run_id: str) -> PublishResult:
        payload = _graph_to_alation_payload(graph, run_id)
        result = PublishResult(
            status="skipped",
            catalog=self.catalog_name,
            run_id=run_id,
            nodes_published=len(graph.nodes),
            edges_published=sum(e.review_status != "rejected" for e in graph.edges),
            payload=payload,
        )

        missing = self.validate_config()
        if missing:
            result.errors = [f"Missing config: {', '.join(missing)}"]
            result.status = "skipped"
            return result

        try:
            client = _AlationClient(settings.alation_base_url, settings.alation_api_token)
            published_urls = []
            errors = []

            for lineage_item in payload["lineage"]:
                try:
                    resp = client.post("/integration/v1/lineage/", lineage_item)
                    url = resp.get("url") or f"{settings.alation_base_url}/lineage/{resp.get('id', '')}"
                    published_urls.append(url)
                except Exception as exc:
                    errors.append(f"Edge {lineage_item['dataflow']['external_id']}: {exc}")

            result.status = "published" if not errors else "error"
            result.catalog_urls = published_urls
            result.errors = errors

        except Exception as exc:
            result.status = "error"
            result.errors = [str(exc)]

        return result


# ---------------------------------------------------------------------------
# Collibra publisher
# ---------------------------------------------------------------------------

class CollibraPublisher(CatalogPublisher):
    """
    Publish lineage to Collibra via the REST API v2.

    Flow:
      1. Upsert each lineage node as a Collibra asset.
      2. Create lineage relations between source and target assets.

    Auth: HTTP Basic (username / password)
    """

    catalog_name = "collibra"

    def validate_config(self) -> list[str]:
        missing = []
        if not settings.collibra_base_url:
            missing.append("COLLIBRA_BASE_URL")
        if not settings.collibra_username:
            missing.append("COLLIBRA_USERNAME")
        if not settings.collibra_password:
            missing.append("COLLIBRA_PASSWORD")
        if not settings.collibra_domain_id:
            missing.append("COLLIBRA_DOMAIN_ID")
        return missing

    def publish(self, graph: LineageGraph, run_id: str) -> PublishResult:
        payload = _graph_to_collibra_payload(graph, run_id)
        result = PublishResult(
            status="skipped",
            catalog=self.catalog_name,
            run_id=run_id,
            nodes_published=len(graph.nodes),
            edges_published=sum(e.review_status != "rejected" for e in graph.edges),
            payload=payload,
        )

        missing = self.validate_config()
        if missing:
            result.errors = [f"Missing config: {', '.join(missing)}"]
            return result

        try:
            client = _CollibraClient(
                settings.collibra_base_url,
                settings.collibra_username,
                settings.collibra_password,
            )
            errors = []
            catalog_urls = []

            # Step 1: upsert assets
            asset_id_map: dict[str, str] = {}  # node.id → collibra asset UUID
            for asset in payload["assets"]:
                try:
                    resp = client.upsert_asset(asset)
                    collibra_id = resp.get("id", asset["external_id"])
                    asset_id_map[asset["external_id"]] = collibra_id
                    url = f"{settings.collibra_base_url}/asset/{collibra_id}"
                    catalog_urls.append(url)
                except Exception as exc:
                    errors.append(f"Asset {asset['name']}: {exc}")

            # Step 2: create relations
            for rel in payload["relations"]:
                src_collibra = asset_id_map.get(rel["source_external_id"])
                tgt_collibra = asset_id_map.get(rel["target_external_id"])
                if not src_collibra or not tgt_collibra:
                    errors.append(
                        f"Skipped relation: asset not found for "
                        f"{rel['source_external_id']} → {rel['target_external_id']}"
                    )
                    continue
                try:
                    client.create_relation({
                        "sourceId":  src_collibra,
                        "targetId":  tgt_collibra,
                        "typeId":    settings.collibra_lineage_type_id,
                        "description": rel.get("description", ""),
                    })
                except Exception as exc:
                    errors.append(f"Relation {src_collibra}→{tgt_collibra}: {exc}")

            result.status = "published" if not errors else "error"
            result.catalog_urls = catalog_urls
            result.errors = errors

        except Exception as exc:
            result.status = "error"
            result.errors = [str(exc)]

        return result


# ---------------------------------------------------------------------------
# Dry-run publisher — formats payload, skips HTTP
# ---------------------------------------------------------------------------

class DryRunPublisher(CatalogPublisher):
    """
    Serializes the graph into the target catalog's payload format and returns it
    without making any HTTP calls. Useful for preview and testing.
    """

    catalog_name = "dry_run"

    def __init__(self, target_catalog: str = "alation") -> None:
        self.target_catalog = target_catalog

    def publish(self, graph: LineageGraph, run_id: str) -> PublishResult:
        if self.target_catalog == "alation":
            payload = _graph_to_alation_payload(graph, run_id)
        elif self.target_catalog == "collibra":
            payload = _graph_to_collibra_payload(graph, run_id)
        else:
            payload = _graph_to_alation_payload(graph, run_id)

        return PublishResult(
            status="dry_run",
            catalog=f"dry_run:{self.target_catalog}",
            run_id=run_id,
            nodes_published=len(graph.nodes),
            edges_published=sum(e.review_status != "rejected" for e in graph.edges),
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Alation payload serialisation
# ---------------------------------------------------------------------------

def _column_mapping_payload(
    mapping: ColumnMapping,
    edge_sme_approved_by: str | None = None,
) -> dict:
    """Serialize a column mapping consistently across catalog payloads."""
    return {
        "source_column": mapping.source_column,
        "target_column": mapping.target_column,
        "transformation": mapping.transformation or "",
        "confidence": mapping.confidence,
        "confidence_level": mapping.confidence_level.value,
        "evidence": mapping.evidence or "",
        # Edge approval covers all mappings unless a mapping has its own
        # explicit approval metadata.
        "sme_approved_by": mapping.sme_approved_by or edge_sme_approved_by or "",
    }


def _graph_to_alation_payload(graph: LineageGraph, run_id: str) -> dict:
    """
    Converts a LineageGraph into an Alation Integration API v1 payload.

    One lineage entry per edge, each describing a dataflow with its
    source (input) and target (output) objects.
    """
    lineage_items = []

    for edge in graph.edges:
        if edge.review_status == "rejected":
            continue
        src = graph.nodes.get(edge.source_id)
        tgt = graph.nodes.get(edge.target_id)
        if not src or not tgt:
            continue

        item = {
            "dataflow": {
                "external_id": f"lineage_agent_{run_id}_{edge.id[:8]}",
                "name": edge.transformation or f"{src.name} → {tgt.name}",
                "description": (
                    f"Confidence: {edge.confidence:.0%}. "
                    f"Evidence: {edge.evidence or 'none'}. "
                    f"Run: {run_id}"
                ),
                "url": settings.alation_lineage_parent_path,
            },
            "source_objects": [_node_to_alation_obj(src)],
            "target_objects": [_node_to_alation_obj(tgt)],
            "column_mappings": [
                _column_mapping_payload(mapping, edge.sme_approved_by)
                for mapping in edge.column_mappings
            ],
        }
        if edge.sme_approved_by:
            item["dataflow"]["description"] += f". SME-approved by: {edge.sme_approved_by}"

        lineage_items.append(item)

    return {
        "run_id": run_id,
        "node_count": len(graph.nodes),
        "edge_count": sum(e.review_status != "rejected" for e in graph.edges),
        "lineage": lineage_items,
    }


def _node_to_alation_obj(node: LineageNode) -> dict:
    return {
        "external_id":          node.id,
        "object_type":          _ALATION_OBJECT_TYPE.get(node.system, "DATAFLOW"),
        "fully_qualified_name": node.full_path,
        "name":                 node.name,
        "description":          node.description or "",
        "url":                  node.full_path,
    }


# ---------------------------------------------------------------------------
# Collibra payload serialisation
# ---------------------------------------------------------------------------

def _graph_to_collibra_payload(graph: LineageGraph, run_id: str) -> dict:
    """
    Converts a LineageGraph into Collibra REST API v2 payloads.

    Returns:
      assets   — one per node, using a deterministic UUID derived from full_path
      relations — one per edge, referencing source/target by external_id
    """
    assets = []
    seen_paths: set[str] = set()

    for node in graph.nodes.values():
        if node.full_path in seen_paths:
            continue
        seen_paths.add(node.full_path)

        # Deterministic UUID so repeat runs don't create duplicate assets
        external_id = str(uuid.uuid5(uuid.NAMESPACE_URL, node.full_path))
        assets.append({
            "external_id":    external_id,
            "node_id":        node.id,
            "name":           node.name,
            "full_path":      node.full_path,
            "asset_type":     _COLLIBRA_ASSET_TYPE_NAME.get(node.system, "Data Asset"),
            "system":         node.system.value,
            "description":    node.description or "",
            "domain_id":      settings.collibra_domain_id,
        })

    # Build a node_id → external_id lookup for relation creation
    node_to_ext: dict[str, str] = {
        a["node_id"]: a["external_id"] for a in assets
    }

    relations = []
    for edge in graph.edges:
        if edge.review_status == "rejected":
            continue
        src_ext = node_to_ext.get(edge.source_id)
        tgt_ext = node_to_ext.get(edge.target_id)
        if not src_ext or not tgt_ext:
            continue

        relations.append({
            "source_external_id": src_ext,
            "target_external_id": tgt_ext,
            "type_id":            settings.collibra_lineage_type_id,
            "description":        (
                f"{edge.transformation or 'lineage'} | "
                f"confidence {edge.confidence:.0%} | "
                f"run {run_id}"
            ),
            "sme_approved_by":    edge.sme_approved_by or "",
            "system_boundary":    edge.system_boundary or "",
            "column_mappings": [
                _column_mapping_payload(mapping, edge.sme_approved_by)
                for mapping in edge.column_mappings
            ],
        })

    return {
        "run_id":   run_id,
        "assets":   assets,
        "relations": relations,
    }


# ---------------------------------------------------------------------------
# HTTP clients (thin wrappers — real calls made only when credentials exist)
# ---------------------------------------------------------------------------

class _AlationClient:
    def __init__(self, base_url: str, token: str) -> None:
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"TOKEN {token}",
            "Content-Type":  "application/json",
        })
        self._base = base_url.rstrip("/")

    def post(self, path: str, payload: dict) -> dict:
        resp = self._session.post(f"{self._base}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()


class _CollibraClient:
    def __init__(self, base_url: str, username: str, password: str) -> None:
        import requests
        self._session = requests.Session()
        self._session.auth = (username, password)
        self._session.headers.update({"Content-Type": "application/json"})
        self._base = base_url.rstrip("/")

    def upsert_asset(self, asset: dict) -> dict:
        """Create or update a Collibra asset by external_id."""
        # Search for existing asset by name + domain
        search = self._session.get(
            f"{self._base}/rest/2.0/assets",
            params={"name": asset["name"], "domainId": asset["domain_id"], "limit": 1},
            timeout=30,
        )
        search.raise_for_status()
        results = search.json().get("results", [])

        if results:
            return results[0]

        # Create new
        resp = self._session.post(
            f"{self._base}/rest/2.0/assets",
            json={
                "name":        asset["name"],
                "domainId":    asset["domain_id"],
                "description": asset.get("description", ""),
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def create_relation(self, relation: dict) -> dict:
        resp = self._session.post(
            f"{self._base}/rest/2.0/relations",
            json=relation,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_publisher(catalog: str) -> CatalogPublisher:
    publishers: dict[str, CatalogPublisher] = {
        "alation":  AlationPublisher(),
        "collibra": CollibraPublisher(),
        "dry_run":  DryRunPublisher("alation"),
    }
    if catalog not in publishers:
        raise ValueError(f"Unknown catalog: '{catalog}'. Choose from: {list(publishers)}")
    return publishers[catalog]
