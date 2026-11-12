from __future__ import annotations

import re
import uuid

import networkx as nx

from src.graph.models import (
    LineageFragment, LineageGraph, LineageNode, LineageEdge, SystemType,
    ProducerClaim, ProducerConflict,
)
from src.graph.paths import (
    adls_display_name, adls_location_key, is_adls_url, is_templated, normalise_location,
)
from src.graph.stewardship import (
    ROOT_SYSTEMS, declared_edge, external_path, make_external_node,
)
from src.parsers.pyspark_ast import PLACEHOLDER_RE

# Systems that legitimately have no upstream data source
_ROOT_SYSTEMS = ROOT_SYSTEMS

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

    # Pass 1b: merge nodes that address the same ADLS location in different
    # shapes (ADF dataset fileSystem+folderPath vs notebook abfss:// URL).
    _merge_storage_locations(graph, path_to_id)

    # Which canonical nodes are evidenced by something other than a catalog
    # inventory export (used to drop unconnected catalog-only tables later).
    evidenced_ids = {
        path_to_id[n.full_path]
        for f in fragments if not f.raw_metadata.get("catalog_only")
        for n in f.nodes
    }

    orchestration_by_fragment = _orchestration_tags(fragments)

    # Pass 2: remap edge endpoints to canonical ids, deduplicate edges
    # edge key: (canonical_source_id, canonical_target_id)
    best_edge: dict[tuple[str, str], LineageEdge] = {}

    for fragment in fragments:
        # Build per-fragment id → canonical id map
        id_map: dict[str, str] = {
            node.id: path_to_id[node.full_path]
            for node in fragment.nodes
        }
        frag_orch = orchestration_by_fragment.get(fragment.id)

        for edge in fragment.edges:
            src_id = id_map.get(edge.source_id, edge.source_id)
            tgt_id = id_map.get(edge.target_id, edge.target_id)

            # Drop edges whose endpoints were never registered as nodes
            if src_id not in graph.nodes or tgt_id not in graph.nodes:
                continue

            key = (src_id, tgt_id)
            canonical = edge.model_copy(update={"source_id": src_id, "target_id": tgt_id})
            if frag_orch and not canonical.orchestrated_by:
                canonical.orchestrated_by = frag_orch
            if not fragment.raw_metadata.get("catalog_only"):
                canonical.produced_by = _union(canonical.produced_by, [fragment.artifact_name])

            existing = best_edge.get(key)
            if existing is None:
                best_edge[key] = canonical
            else:
                merged = _merge_column_mappings([existing, canonical])
                producers = _union(existing.produced_by, canonical.produced_by)
                # Different artifacts run under different orchestrators: an
                # edge only carries a tag when every producer agrees on it.
                orch = existing.orchestrated_by if existing.orchestrated_by == canonical.orchestrated_by else None
                if canonical.confidence > existing.confidence:
                    canonical.column_mappings = merged
                    canonical.produced_by = producers
                    canonical.orchestrated_by = orch
                    best_edge[key] = canonical
                else:
                    existing.column_mappings = merged
                    existing.produced_by = producers
                    existing.orchestrated_by = orch

    graph.edges = list(best_edge.values())

    # Pass 3: resolve templated names (f-strings, parameterised ADF paths)
    # against concrete nodes, e.g. pos_{country}_transactions → pos_nl_transactions.
    created = _expand_templates(graph)
    evidenced_ids |= created

    _drop_unconnected_catalog_nodes(graph, evidenced_ids)
    _apply_catalog_declared_producers(graph, fragments)
    graph.producer_conflicts = _detect_producer_conflicts(
        graph, fragments, orchestration_by_fragment,
    )
    _annotate_system_boundaries(graph)
    return graph


def _apply_catalog_declared_producers(graph: LineageGraph, fragments: list[LineageFragment]) -> None:
    """
    Add ``external → table`` edges for catalog tables declared as externally
    produced — but only for tables already in the graph and with no producing
    code found (code evidence beats a declaration).
    """
    declared: dict[str, tuple[str, str]] = {}
    for f in fragments:
        for full_name, producer in (f.raw_metadata.get("declared_producers") or {}).items():
            declared.setdefault(full_name, (producer, f.artifact_name))
    if not declared:
        return
    by_path = {n.full_path: n for n in graph.nodes.values()}
    has_incoming = {e.target_id for e in graph.edges}
    for full_name, (producer, artifact) in declared.items():
        tgt = by_path.get(full_name)
        if tgt is None or tgt.id in has_incoming:
            continue
        ext = by_path.get(external_path(producer))
        if ext is None:
            ext = make_external_node(producer, description=f"declared in {artifact}")
            graph.nodes[ext.id] = ext
            by_path[ext.full_path] = ext
        edge = declared_edge(ext, tgt, f"table comment in {artifact}")
        edge.produced_by = [artifact]
        graph.edges.append(edge)


def _union(a: list[str], b: list[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


# Artifact types whose edges represent code that writes the target. ADF copy
# fragments are excluded: several pipelines landing files in one ADLS folder
# is normal and not a "which version is live" question.
_PRODUCER_TYPES = {"notebook", "dlt", "sql"}


def _detect_producer_conflicts(
    graph: LineageGraph,
    fragments: list[LineageFragment],
    orchestration_by_fragment: dict[str, str],
) -> list[ProducerConflict]:
    """
    Flag targets written by more than one artifact (e.g. margin_v1 and
    margin_v2 both ``saveAsTable("gold.fact_store_week_sales")``).

    Evidence gathered per claim — the agent presents it but does not pick:
    * which ADF activity (if any) runs the artifact;
    * whether the Unity Catalog ``last_write.notebook_path`` names it;
    * which inputs it reads (claims that differ in inputs change lineage).
    """
    frag_by_name = {f.artifact_name: f for f in fragments}
    last_write: dict[str, dict] = {}
    for f in fragments:
        last_write.update(f.raw_metadata.get("last_write") or {})

    producers_by_target: dict[str, dict[str, list[str]]] = {}
    for e in graph.edges:
        for art in e.produced_by:
            frag = frag_by_name.get(art)
            if frag is None or frag.artifact_type not in _PRODUCER_TYPES:
                continue
            producers_by_target.setdefault(e.target_id, {}).setdefault(art, []).append(e.source_id)

    conflicts: list[ProducerConflict] = []
    for target_id, claims_map in producers_by_target.items():
        if len(claims_map) < 2:
            continue
        target = graph.nodes.get(target_id)
        lw = last_write.get(target.full_path) if target else None
        lw_path = _notebook_key((lw or {}).get("notebook_path"))
        claims = []
        for art, src_ids in claims_map.items():
            frag = frag_by_name[art]
            nb_path = frag.raw_metadata.get("notebook_path")
            claims.append(ProducerClaim(
                artifact=art,
                notebook_path=nb_path,
                orchestrated_by=orchestration_by_fragment.get(frag.id),
                source_ids=list(dict.fromkeys(src_ids)),
                matches_catalog_last_write=bool(lw_path and _notebook_key(nb_path) == lw_path),
            ))
        conflicts.append(ProducerConflict(
            target_id=target_id, claims=claims, catalog_last_write=lw,
        ))
    return conflicts


# ---------------------------------------------------------------------------
# Cross-system helpers
# ---------------------------------------------------------------------------

def _location_key(node: LineageNode) -> str | None:
    if is_adls_url(node.full_path):
        return adls_location_key(node.full_path)
    if node.full_path.startswith(_ADF_NS) and node.system == SystemType.ADLS and node.description:
        desc = node.description.strip()
        if desc.startswith(("abfss://", "abfs://", "wasbs://")):
            return adls_location_key(desc)
        if "/" in desc:
            return normalise_location(desc)
    return None


def _merge_storage_locations(graph: LineageGraph, path_to_id: dict[str, str]) -> None:
    """
    Collapse nodes sharing an ADLS location key into one canonical node. The
    concrete ``abfss://`` path wins as full_path; an ADF dataset name, if any,
    is kept as the display name since that is what data engineers recognise.
    """
    groups: dict[str, list[str]] = {}
    for nid, node in graph.nodes.items():
        key = _location_key(node)
        if key:
            groups.setdefault(key, []).append(nid)

    for key, ids in groups.items():
        if len(ids) < 2:
            continue
        nodes = [graph.nodes[i] for i in ids]
        keep = next((n for n in nodes if is_adls_url(n.full_path)), nodes[0])
        dataset = next((n for n in nodes if n.full_path.startswith(_ADF_NS)), None)
        for other in nodes:
            if other.id == keep.id:
                continue
            _merge_node_metadata(keep, other)
            for p, i in list(path_to_id.items()):
                if i == other.id:
                    path_to_id[p] = keep.id
            del graph.nodes[other.id]
        keep.system = SystemType.ADLS
        if dataset is not None:
            keep.name = dataset.name
            loc = adls_display_name(keep.full_path)
            keep.description = f"{loc} (ADF dataset {dataset.name})"


def _notebook_key(path: str | None) -> str | None:
    if not path:
        return None
    return path.rstrip("/").lower()


def _orchestration_tags(fragments: list[LineageFragment]) -> dict[str, str]:
    """
    Map notebook fragment id → ``adf:<pipeline>/<activity>`` for every notebook
    that an ADF DatabricksNotebook activity triggers. Matching uses the notebook
    path recorded in the notebook source; failing that, the trailing path
    segments against the uploaded filename (``.../bronze/pos_ingest`` ↔
    ``databricks_bronze_pos_ingest.py``).
    """
    by_path: dict[str, str] = {}
    for f in fragments:
        for o in f.raw_metadata.get("orchestrations", []) or []:
            key = _notebook_key(o.get("notebook_path"))
            if key:
                by_path.setdefault(key, f"adf:{o.get('pipeline')}/{o.get('activity')}")

    if not by_path:
        return {}

    out: dict[str, str] = {}
    for f in fragments:
        if f.artifact_type not in ("notebook", "dlt"):
            continue
        key = _notebook_key(f.raw_metadata.get("notebook_path"))
        if key and key in by_path:
            out[f.id] = by_path[key]
            continue
        stem = re.sub(r"\.(py|ipynb|sql|scala)$", "", f.artifact_name.lower())
        for path, tag in by_path.items():
            tail = "_".join(path.strip("/").split("/")[-2:])
            if tail and (stem == tail or stem.endswith("_" + tail)):
                out[f.id] = tag
                break
    return out


def _template_regex(template: str) -> re.Pattern:
    parts, last, seen = [], 0, {}
    for m in PLACEHOLDER_RE.finditer(template):
        parts.append(re.escape(template[last:m.start()]))
        name = m.group(1)
        if name in seen:
            parts.append(f"(?P={seen[name]})")
        else:
            group = f"g{len(seen)}_{name}"
            seen[name] = group
            parts.append(f"(?P<{group}>[^./]+)")
        last = m.end()
    parts.append(re.escape(template[last:]))
    return re.compile("".join(parts) + r"/?", re.IGNORECASE)


def _bindings(match: re.Match) -> dict[str, str]:
    return {g.split("_", 1)[1]: v for g, v in match.groupdict().items()}


def _substitute(template: str, binding: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        val = binding.get(m.group(1))
        if val is None:
            return m.group(0)
        mode = m.group(2)
        return val.upper() if mode == "upper" else val.lower() if mode == "lower" else val
    return PLACEHOLDER_RE.sub(repl, template)


def _expand_templates(graph: LineageGraph) -> set[str]:
    """
    Replace edges touching a templated node with concrete per-instance edges.

    * Templated ↔ concrete: one edge per concrete node matching the template.
    * Templated ↔ templated (e.g. ``raw/pos/{country:upper}`` →
      ``bronze.pos_{country:lower}_transactions``): bindings come from whichever
      side matches concrete nodes and are substituted into the other side,
      creating the concrete counterpart node if it was never seen.

    Expanded edges are tagged ``pattern match`` in evidence so scoring keeps
    them at medium confidence until confirmed. Returns ids of created nodes.
    """
    created: set[str] = set()
    for _ in range(4):  # fixpoint: newly created concrete nodes can unlock more
        templated = {nid for nid, n in graph.nodes.items() if is_templated(n.full_path)}
        if not templated:
            break
        concrete = [n for n in graph.nodes.values() if not is_templated(n.full_path)]

        def matches(nid: str) -> list[tuple[str, dict[str, str]]]:
            rx = _template_regex(graph.nodes[nid].full_path)
            out = []
            for c in concrete:
                m = rx.fullmatch(c.full_path)
                if m:
                    out.append((c.id, _bindings(m)))
            return out

        match_cache = {nid: matches(nid) for nid in templated}
        new_edges: list[LineageEdge] = []
        changed = False
        existing_keys = {(e.source_id, e.target_id) for e in graph.edges}

        for edge in graph.edges:
            s_t, t_t = edge.source_id in templated, edge.target_id in templated
            if not (s_t or t_t):
                new_edges.append(edge)
                continue

            pairs: list[tuple[str, str, dict[str, str]]] = []
            if s_t and t_t:
                src_m, tgt_m = match_cache[edge.source_id], match_cache[edge.target_id]
                if tgt_m:
                    for tid, b in tgt_m:
                        sid = _concrete_for(graph, edge.source_id, b, created)
                        if sid:
                            pairs.append((sid, tid, b))
                elif src_m:
                    for sid, b in src_m:
                        tid = _concrete_for(graph, edge.target_id, b, created)
                        if tid:
                            pairs.append((sid, tid, b))
            elif s_t:
                pairs = [(sid, edge.target_id, b) for sid, b in match_cache[edge.source_id]]
            else:
                pairs = [(edge.source_id, tid, b) for tid, b in match_cache[edge.target_id]]

            if not pairs:
                new_edges.append(edge)  # unresolved — stays as a templated gap
                continue

            changed = True
            for sid, tid, b in pairs:
                if (sid, tid) in existing_keys:
                    continue
                existing_keys.add((sid, tid))
                desc = ", ".join(f"{k}={v}" for k, v in b.items())
                new_edges.append(edge.model_copy(update={
                    "id": str(uuid.uuid4()),
                    "source_id": sid,
                    "target_id": tid,
                    "evidence": f"{edge.evidence or ''} (pattern match: {desc})".strip(),
                    "column_mappings": list(edge.column_mappings),
                }))

        graph.edges = new_edges
        # Remove templated nodes that no longer carry any edge.
        used = {e.source_id for e in graph.edges} | {e.target_id for e in graph.edges}
        for nid in templated:
            if nid not in used and nid in graph.nodes:
                del graph.nodes[nid]
        if not changed:
            break
    return created


def _concrete_for(graph: LineageGraph, template_id: str, binding: dict[str, str], created: set[str]) -> str | None:
    tmpl = graph.nodes[template_id]
    path = _substitute(tmpl.full_path, binding)
    if is_templated(path):
        return None
    for nid, n in graph.nodes.items():
        if n.full_path == path:
            return nid
    name = adls_display_name(path) if is_adls_url(path) else path.split(".")[-1]
    node = LineageNode(
        name=name, system=tmpl.system, full_path=path,
        description=_substitute(tmpl.description, binding) if tmpl.description else None,
        owner=tmpl.owner,
    )
    graph.nodes[node.id] = node
    created.add(node.id)
    return node.id


def _drop_unconnected_catalog_nodes(graph: LineageGraph, evidenced_ids: set[str]) -> None:
    """Catalog inventory tables only appear in the graph once lineage touches them."""
    used = {e.source_id for e in graph.edges} | {e.target_id for e in graph.edges}
    for nid in list(graph.nodes):
        if nid not in used and nid not in evidenced_ids:
            del graph.nodes[nid]


def _merge_node_metadata(existing: LineageNode, incoming: LineageNode) -> None:
    """
    Update an existing node with richer metadata from a later fragment.
    Priority: prefer non-UNKNOWN system type, non-empty description/owner.
    """
    if existing.system == SystemType.UNKNOWN and incoming.system != SystemType.UNKNOWN:
        existing.system = incoming.system
    # Legacy "adf" placeholders: ADF is an orchestrator, so any concrete store
    # type from a dataset definition supersedes it.
    elif existing.system == SystemType.ADF and incoming.system not in (SystemType.UNKNOWN, SystemType.ADF):
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
