from __future__ import annotations

import networkx as nx

from src.graph.models import LineageGraph, LineageNode, LineageEdge, Gap, SystemType
from src.graph.stitcher import to_networkx, _ADF_NS, _ROOT_SYSTEMS

# Confidence threshold below which an edge is flagged as needing SME review
_LOW_CONFIDENCE = 0.70

# Systems that are valid as terminal sinks (no required downstream)
_SINK_SYSTEMS = {SystemType.BI}


def detect_gaps(graph: LineageGraph) -> list[Gap]:
    """
    Scan the stitched lineage graph for gaps that need SME input.

    Gap types detected:
    - unresolved_dataset_ref: adf:// node still present — pipeline references a
      dataset whose JSON was never uploaded
    - unknown_system:         node whose system type is still UNKNOWN
    - missing_source:         non-root node has no incoming edges
    - missing_target:         non-sink node has no outgoing edges
    - low_confidence_edge:    edge (especially cross-system) with confidence < threshold
    - contradictory_paths:    two edges with the same (source, target) pair but
                              different transformation descriptions
    - isolated_subgraph:      connected component with no ADLS or known root node
                              (entire sub-pipeline has unclear origin)
    """
    gaps: list[Gap] = []
    G = to_networkx(graph)

    gaps.extend(_unresolved_adf_refs(graph))
    gaps.extend(_unknown_system_nodes(graph))
    gaps.extend(_missing_sources(graph, G))
    gaps.extend(_missing_targets(graph, G))
    gaps.extend(_low_confidence_edges(graph))
    gaps.extend(_contradictory_paths(graph))
    gaps.extend(_isolated_subgraphs(graph, G))

    return gaps


# ---------------------------------------------------------------------------
# Individual gap detectors
# ---------------------------------------------------------------------------

def _unresolved_adf_refs(graph: LineageGraph) -> list[Gap]:
    """
    ADF pipeline parsers use adf://DatasetName as a placeholder.
    If a dataset JSON was also uploaded the stitcher merges the node and
    resolves its system type. Nodes that still carry the adf:// prefix after
    stitching mean no dataset JSON was uploaded for that reference.
    """
    gaps = []
    for node_id, node in graph.nodes.items():
        if node.full_path.startswith(_ADF_NS):
            dataset_name = node.full_path[len(_ADF_NS):]
            gaps.append(Gap(
                gap_type="unresolved_dataset_ref",
                node_id=node_id,
                description=(
                    f"ADF dataset '{dataset_name}' is referenced in a pipeline but "
                    f"no dataset JSON was uploaded for it. "
                    f"Upload the dataset JSON or provide the actual storage path."
                ),
            ))
    return gaps


def _unknown_system_nodes(graph: LineageGraph) -> list[Gap]:
    """Nodes whose system type could not be inferred need SME confirmation."""
    gaps = []
    for node_id, node in graph.nodes.items():
        if node.system == SystemType.UNKNOWN and not node.full_path.startswith(_ADF_NS):
            gaps.append(Gap(
                gap_type="unknown_system",
                node_id=node_id,
                description=(
                    f"Could not determine which system '{node.name}' belongs to "
                    f"(path: {node.full_path}). "
                    f"Please confirm: is this ADLS, Databricks, SQL, or another system?"
                ),
            ))
    return gaps


def _missing_sources(graph: LineageGraph, G) -> list[Gap]:
    """
    A non-root node with in-degree 0 has no known upstream data source.
    Root systems (ADLS) are intentionally source-less.
    """
    gaps = []
    for node_id, node in graph.nodes.items():
        if node.system in _ROOT_SYSTEMS:
            continue
        if node.full_path.startswith(_ADF_NS):
            continue  # already flagged by _unresolved_adf_refs
        if G.in_degree(node_id) == 0:
            gaps.append(Gap(
                gap_type="missing_source",
                node_id=node_id,
                description=(
                    f"'{node.name}' ({node.system.value}) has no upstream source in the lineage graph. "
                    f"Where does this data come from?"
                ),
            ))
    return gaps


def _missing_targets(graph: LineageGraph, G) -> list[Gap]:
    """
    A non-sink node with out-degree 0 may indicate an incomplete pipeline —
    data is loaded somewhere but we have no record of it being consumed.
    Only flag when there is more than one node (a graph with a single node
    trivially has out-degree 0).
    """
    if len(graph.nodes) <= 1:
        return []

    gaps = []
    for node_id, node in graph.nodes.items():
        if node.system in _SINK_SYSTEMS:
            continue
        if node.full_path.startswith(_ADF_NS):
            continue
        if G.out_degree(node_id) == 0 and G.in_degree(node_id) > 0:
            gaps.append(Gap(
                gap_type="missing_target",
                node_id=node_id,
                description=(
                    f"'{node.name}' ({node.system.value}) has no downstream consumers in the lineage graph. "
                    f"Is this a final output, or does it feed into another system not yet documented?"
                ),
            ))
    return gaps


def _low_confidence_edges(graph: LineageGraph) -> list[Gap]:
    """
    Edges below the confidence threshold (especially cross-system ones)
    should be confirmed by an SME before being published to the catalog.
    """
    gaps = []
    for edge in graph.edges:
        if edge.confidence < _LOW_CONFIDENCE:
            boundary_note = f" (crosses {edge.system_boundary})" if edge.system_boundary else ""
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            src_name = src.name if src else edge.source_id
            tgt_name = tgt.name if tgt else edge.target_id
            gaps.append(Gap(
                gap_type="low_confidence_edge",
                edge_id=edge.id,
                description=(
                    f"Edge from '{src_name}' → '{tgt_name}'{boundary_note} "
                    f"has confidence {edge.confidence:.0%} (below {_LOW_CONFIDENCE:.0%} threshold). "
                    f"Evidence: {edge.evidence or 'none'}. "
                    f"Please confirm this relationship is correct."
                ),
            ))
    return gaps


def _contradictory_paths(graph: LineageGraph) -> list[Gap]:
    """
    Multiple edges between the same (source, target) pair with different
    transformation descriptions suggest conflicting documentation.
    """
    # Group edges by (source_id, target_id)
    seen: dict[tuple[str, str], list[LineageEdge]] = {}
    for edge in graph.edges:
        key = (edge.source_id, edge.target_id)
        seen.setdefault(key, []).append(edge)

    gaps = []
    for (src_id, tgt_id), edges in seen.items():
        if len(edges) < 2:
            continue
        transforms = {e.transformation for e in edges if e.transformation}
        if len(transforms) > 1:
            src = graph.nodes.get(src_id)
            tgt = graph.nodes.get(tgt_id)
            gaps.append(Gap(
                gap_type="contradictory_paths",
                edge_id=edges[0].id,
                description=(
                    f"Multiple conflicting transformations documented for "
                    f"'{src.name if src else src_id}' → '{tgt.name if tgt else tgt_id}': "
                    f"{sorted(transforms)}. Which is correct?"
                ),
            ))
    return gaps


def _isolated_subgraphs(graph: LineageGraph, G) -> list[Gap]:
    """
    A weakly-connected component where no node belongs to a root system
    (i.e. the entire sub-pipeline has no known origin) is suspicious —
    it likely represents an incomplete lineage capture.
    Flag only components with 2+ nodes (single isolated nodes are caught
    by _missing_sources).
    """
    gaps = []
    for component in nx.weakly_connected_components(G):
        if len(component) < 2:
            continue
        node_systems = {graph.nodes[n].system for n in component if n in graph.nodes}
        has_root = bool(node_systems & _ROOT_SYSTEMS)
        has_adf_placeholder = any(
            graph.nodes[n].full_path.startswith(_ADF_NS)
            for n in component if n in graph.nodes
        )
        if not has_root and not has_adf_placeholder:
            sample_names = [
                graph.nodes[n].name for n in list(component)[:3] if n in graph.nodes
            ]
            gaps.append(Gap(
                gap_type="isolated_subgraph",
                description=(
                    f"A group of {len(component)} nodes ({', '.join(sample_names)}, ...) "
                    f"has no connection to a known data source (e.g. ADLS). "
                    f"What is the origin of this data?"
                ),
            ))
    return gaps
