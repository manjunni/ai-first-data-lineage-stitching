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

    # Pass 1: merge nodes, building the path→id registry
    for fragment in fragments:
        for node in fragment.nodes:
            if node.full_path in path_to_id:
                existing_id = path_to_id[node.full_path]
                _merge_node_metadata(graph.nodes[existing_id], node)
            else:
                path_to_id[node.full_path] = node.id
                graph.nodes[node.id] = node

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
            if existing is None or canonical.confidence > existing.confidence:
                best_edge[key] = canonical

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
