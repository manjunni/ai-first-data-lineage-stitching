from __future__ import annotations

import networkx as nx

from src.graph.models import (
    LineageFragment, LineageGraph, LineageNode, LineageEdge, SystemType,
)

# Systems that legitimately have no upstream data source
_ROOT_SYSTEMS = {SystemType.ADLS}

# Namespace prefix from ADF parser — nodes that still carry this are unresolved
_ADF_NS = "adf://"


def stitch_fragments(fragments: list[LineageFragment]) -> LineageGraph:
    """
    Merge all parsed fragments into a single lineage graph.

    Deduplication rules:
    - Nodes: matched by full_path. When the same path appears in multiple
      fragments the richer node wins (system type, description, owner).
    - Edges: deduplicated by (source_id, target_id) pair. When duplicates
      exist the one with the higher confidence is kept.

    Post-processing:
    - System boundaries are annotated on every cross-system edge.
    - Dangling edge endpoints (IDs not in node registry) are removed.
    """
    graph = LineageGraph()
    # full_path → canonical node id
    path_to_id: dict[str, str] = {}

    # Pass 1: merge nodes, building the path→id registry.
    # Secondary index by last path segment to merge unqualified names
    # (e.g. "customer_orders") with schema-qualified ones
    # (e.g. "curated.customer_orders") when they refer to the same table.
    # Only applied for non-ADLS, non-ADF nodes with a 1-part unqualified path.
    unqualified_to_id: dict[str, str] = {}  # last_segment → node_id (1-part paths only)

    for fragment in fragments:
        for node in fragment.nodes:
            if node.full_path in path_to_id:
                existing_id = path_to_id[node.full_path]
                existing_node = graph.nodes[existing_id]
                _merge_node_metadata(existing_node, node)
                # If the existing node is an adf:// placeholder and the incoming
                # node carries the actual table path, upgrade to the real path.
                if existing_node.full_path.startswith(_ADF_NS) and not node.full_path.startswith(_ADF_NS):
                    path_to_id[node.full_path] = existing_id
                    existing_node.full_path = node.full_path
            else:
                is_storage_path = node.full_path.startswith(("abfss://", "wasbs://", _ADF_NS))
                last_seg = node.full_path.split(".")[-1]
                is_unqualified = "." not in node.full_path

                if not is_storage_path and is_unqualified and last_seg in unqualified_to_id:
                    # Incoming qualified node matches an existing unqualified one —
                    # can't tell which is canonical here; keep existing
                    existing_id = unqualified_to_id[last_seg]
                    path_to_id[node.full_path] = existing_id
                    _merge_node_metadata(graph.nodes[existing_id], node)
                elif not is_storage_path and not is_unqualified and last_seg in unqualified_to_id:
                    # Incoming qualified node whose base name matches an existing
                    # unqualified node — merge and upgrade to the qualified path
                    existing_id = unqualified_to_id[last_seg]
                    existing_node = graph.nodes[existing_id]
                    path_to_id[node.full_path] = existing_id
                    path_to_id[existing_node.full_path] = existing_id
                    existing_node.full_path = node.full_path  # prefer qualified
                    _merge_node_metadata(existing_node, node)
                else:
                    path_to_id[node.full_path] = node.id
                    graph.nodes[node.id] = node
                    if not is_storage_path and is_unqualified:
                        unqualified_to_id[last_seg] = node.id
                    # For ADF dataset nodes whose description is a table reference
                    # (e.g. "bronze.customers"), link them to the actual table node
                    # so notebooks referencing the same table by name are merged.
                    if node.full_path.startswith(_ADF_NS) and node.description:
                        desc = node.description.strip()
                        if desc and "/" not in desc and not desc.startswith("abfss://"):
                            if desc in path_to_id:
                                # Table node was registered first — redirect this
                                # ADF placeholder to the existing canonical node.
                                existing_id = path_to_id[desc]
                                path_to_id[node.full_path] = existing_id
                                del graph.nodes[node.id]
                                _merge_node_metadata(graph.nodes[existing_id], node)
                            else:
                                # ADF node registered first — set up aliases so
                                # the table node merges in when it arrives.
                                path_to_id.setdefault(desc, node.id)
                                seg = desc.split(".")[-1]
                                unqualified_to_id.setdefault(seg, node.id)

    # Pass 2: remap edge endpoints to canonical ids, deduplicate edges
    # edge key: (canonical_source_id, canonical_target_id)
    best_edge: dict[tuple[str, str], LineageEdge] = {}

    for fragment in fragments:
        # Build per-fragment id → canonical id map
        id_map: dict[str, str] = {
            node.id: path_to_id[node.full_path]
            for node in fragment.nodes
        }

        for edge in fragment.edges:
            src_id = id_map.get(edge.source_id, edge.source_id)
            tgt_id = id_map.get(edge.target_id, edge.target_id)

            # Drop edges whose endpoints were never registered as nodes
            if src_id not in graph.nodes or tgt_id not in graph.nodes:
                continue

            key = (src_id, tgt_id)
            canonical = edge.model_copy(update={"source_id": src_id, "target_id": tgt_id})

            existing = best_edge.get(key)
            if existing is None:
                best_edge[key] = canonical
            else:
                merged = _merge_column_mappings([existing, canonical])
                if canonical.confidence > existing.confidence:
                    canonical.column_mappings = merged
                    best_edge[key] = canonical
                else:
                    existing.column_mappings = merged

    graph.edges = list(best_edge.values())

    _annotate_system_boundaries(graph)
    return graph


def _merge_node_metadata(existing: LineageNode, incoming: LineageNode) -> None:
    """
    Update an existing node with richer metadata from a later fragment.
    Priority: prefer non-UNKNOWN system type, non-empty description/owner.
    """
    if existing.system == SystemType.UNKNOWN and incoming.system != SystemType.UNKNOWN:
        existing.system = incoming.system
    if not existing.description and incoming.description:
        existing.description = incoming.description
    if not existing.owner and incoming.owner:
        existing.owner = incoming.owner


def _annotate_system_boundaries(graph: LineageGraph) -> None:
    """Tag every cross-system edge with a human-readable boundary label."""
    for edge in graph.edges:
        src = graph.nodes.get(edge.source_id)
        tgt = graph.nodes.get(edge.target_id)
        if src and tgt and src.system != tgt.system:
            edge.system_boundary = f"{src.system.value}→{tgt.system.value}"
        else:
            edge.system_boundary = None


def _merge_column_mappings(edges: list[LineageEdge]) -> list:
    """Keep the strongest evidence for each source/target column pair."""
    best = {}
    for edge in edges:
        for mapping in edge.column_mappings:
            key = (mapping.source_column, mapping.target_column)
            if key not in best or mapping.confidence > best[key].confidence:
                best[key] = mapping
    return list(best.values())


def to_networkx(graph: LineageGraph) -> nx.DiGraph:
    """Convert to a NetworkX DiGraph for graph-algorithm use."""
    G = nx.DiGraph()
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, **node.model_dump())
    for edge in graph.edges:
        # Guard: only add edges whose endpoints are present
        if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
            G.add_edge(edge.source_id, edge.target_id, **edge.model_dump())
    return G
