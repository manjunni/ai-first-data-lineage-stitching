from __future__ import annotations

from collections import defaultdict

from src.graph.models import LineageGraph, LineageNode, LineageEdge, ConfidenceLevel, SystemType, Gap

_ADF_NS = "adf://"

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Node fill colours by system type (hex)
_SYSTEM_COLOUR: dict[SystemType, tuple[str, str]] = {
    # (fill, font)
    SystemType.ADLS:        ("#4472C4", "#ffffff"),
    SystemType.DATABRICKS:  ("#FF8C00", "#ffffff"),
    SystemType.SQL:         ("#70AD47", "#ffffff"),
    SystemType.DLT:         ("#7030A0", "#ffffff"),
    SystemType.ADF:         ("#ED7D31", "#ffffff"),
    SystemType.BI:          ("#FFC000", "#000000"),
    SystemType.UNKNOWN:     ("#A6A6A6", "#ffffff"),
}

# Cluster background colours by system type
_CLUSTER_BG: dict[SystemType, str] = {
    SystemType.ADLS:        "#EBF3FB",
    SystemType.DATABRICKS:  "#FFF3E6",
    SystemType.SQL:         "#EEF7EC",
    SystemType.DLT:         "#F5EEF8",
    SystemType.ADF:         "#FEF1E8",
    SystemType.BI:          "#FFFBE6",
    SystemType.UNKNOWN:     "#F5F5F5",
}

# Edge colour by confidence level
_EDGE_COLOUR: dict[ConfidenceLevel, str] = {
    ConfidenceLevel.CONFIRMED: "#4472C4",   # blue   — SME-approved
    ConfidenceLevel.HIGH:      "#70AD47",   # green  — high confidence
    ConfidenceLevel.MEDIUM:    "#FFC000",   # amber  — medium confidence
    ConfidenceLevel.LOW:       "#FF0000",   # red    — low / needs review
}

# Mermaid CSS class names per system
_MERMAID_CLASS: dict[SystemType, str] = {
    SystemType.ADLS:        "adls",
    SystemType.DATABRICKS:  "databricks",
    SystemType.SQL:         "sql",
    SystemType.DLT:         "dlt",
    SystemType.ADF:         "adf",
    SystemType.BI:          "bi",
    SystemType.UNKNOWN:     "unknown",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_mermaid(
    graph: LineageGraph,
    gaps: list[Gap] | None = None,
) -> str:
    """
    Generate a Mermaid flowchart with:
    - Nodes grouped into subgraphs by system type
    - Node style classes by system type
    - Edge labels showing transformation + confidence %
    - Edge style (solid / dashed / dotted) by confidence level
    - Gap-flagged nodes highlighted in red
    """
    gap_node_ids: set[str] = {g.node_id for g in (gaps or []) if g.node_id}
    gap_edge_ids: set[str] = {g.edge_id for g in (gaps or []) if g.edge_id}

    lines: list[str] = ["flowchart LR"]

    # Class definitions
    lines.append("")
    for system, css_class in _MERMAID_CLASS.items():
        fill, font = _SYSTEM_COLOUR[system]
        lines.append(f"    classDef {css_class} fill:{fill},color:{font},stroke:#333,stroke-width:1px")
    lines.append("    classDef gap fill:#FF0000,color:#ffffff,stroke:#990000,stroke-width:2px")

    # Group nodes by system into subgraphs
    by_system: dict[SystemType, list[tuple[str, LineageNode]]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        by_system[node.system].append((node_id, node))

    lines.append("")
    for system, node_list in sorted(by_system.items(), key=lambda x: x[0].value):
        if len(by_system) > 1:
            lines.append(f"    subgraph {system.value.upper()}")
        for node_id, node in node_list:
            safe = _safe_id(node_id)
            label = _display_label(node).replace('"', "'")
            lines.append(f'        {safe}["{label}"]')
        if len(by_system) > 1:
            lines.append("    end")

    # Edges
    lines.append("")
    edge_index = 0
    edge_styles: list[str] = []

    for edge in graph.edges:
        src = _safe_id(edge.source_id)
        tgt = _safe_id(edge.target_id)
        label = _edge_label(edge, edge.id in gap_edge_ids)
        style = _mermaid_arrow(edge)
        lines.append(f"    {src} {style}|{label}| {tgt}")
        edge_styles.append(_mermaid_link_style(edge_index, edge, edge.id in gap_edge_ids))
        edge_index += 1

    # Link styles (applied after edges by index)
    if edge_styles:
        lines.append("")
        lines.extend(edge_styles)

    # Apply gap class to flagged nodes
    lines.append("")
    for node_id, node in graph.nodes.items():
        safe = _safe_id(node_id)
        css = "gap" if node_id in gap_node_ids else _MERMAID_CLASS[node.system]
        lines.append(f"    class {safe} {css}")

    return "\n".join(lines)


def generate_graphviz_dot(
    graph: LineageGraph,
    gaps: list[Gap] | None = None,
) -> str:
    """
    Generate a Graphviz DOT file with:
    - Nodes clustered by system type (subgraph cluster_*)
    - Node fill colour by system type
    - Edge colour + style (solid/dashed/dotted) by confidence level
    - Edge label: transformation + confidence %
    - Gap nodes outlined in red with dashed border
    """
    gap_node_ids: set[str] = {g.node_id for g in (gaps or []) if g.node_id}
    gap_edge_ids: set[str] = {g.edge_id for g in (gaps or []) if g.edge_id}

    lines: list[str] = [
        "digraph lineage {",
        "    rankdir=LR;",
        "    compound=true;",
        '    graph [fontname="Helvetica", fontsize=11];',
        '    node  [fontname="Helvetica", fontsize=10, shape=box, style=filled];',
        '    edge  [fontname="Helvetica", fontsize=9];',
        "",
    ]

    # Cluster nodes by system
    by_system: dict[SystemType, list[tuple[str, LineageNode]]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        by_system[node.system].append((node_id, node))

    for system, node_list in sorted(by_system.items(), key=lambda x: x[0].value):
        fill_colour, _ = _SYSTEM_COLOUR[system]
        bg_colour = _CLUSTER_BG[system]
        lines.append(f"    subgraph cluster_{system.value} {{")
        lines.append(f'        label="{system.value.upper()}";')
        lines.append(f'        style=filled; fillcolor="{bg_colour}";')
        lines.append('        fontname="Helvetica-Bold"; fontsize=11;')

        for node_id, node in node_list:
            safe = _safe_id(node_id)
            label = _display_label(node).replace('"', "'").replace("\n", "\\n")
            node_fill = "#FF3333" if node_id in gap_node_ids else fill_colour
            _, font_colour = _SYSTEM_COLOUR[node.system]
            border = 'style="filled,dashed" penwidth=2' if node_id in gap_node_ids else "style=filled"
            lines.append(
                f'        {safe} [label="{label}", fillcolor="{node_fill}", '
                f'fontcolor="{font_colour}", {border}];'
            )

        lines.append("    }")
        lines.append("")

    # Edges
    for edge in graph.edges:
        src = _safe_id(edge.source_id)
        tgt = _safe_id(edge.target_id)
        label = _edge_label(edge, edge.id in gap_edge_ids).replace('"', "'")
        colour = "#FF3333" if edge.id in gap_edge_ids else _EDGE_COLOUR[edge.confidence_level]
        dot_style = _dot_edge_style(edge, edge.id in gap_edge_ids)
        lines.append(
            f'    {src} -> {tgt} [label="{label}", color="{colour}", {dot_style}];'
        )

    lines.append("}")
    return "\n".join(lines)


def generate_summary_html(
    graph: LineageGraph,
    summary: dict,
    gaps: list[Gap] | None = None,
) -> str:
    """
    Generate a minimal HTML snippet with a legend and summary table.
    Suitable for embedding in a web report or Jupyter notebook.
    """
    gap_count = len(gaps) if gaps else 0

    rows = [
        ("Total nodes", len(graph.nodes)),
        ("Total edges", summary.get("total_edges", 0)),
        ("Average confidence", f"{summary.get('avg_confidence', 0):.1%}"
         if summary.get('avg_confidence') is not None else "—"),
        ("SME-confirmed edges", summary.get("sme_confirmed", 0)),
        ("Cross-system edges", summary.get("cross_system_edges", 0)),
        ("Open gaps", gap_count),
    ]

    table_rows = "\n".join(
        f"    <tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows
    )

    level_rows = "\n".join(
        f"    <tr><td>{level}</td><td>{count}</td></tr>"
        for level, count in (summary.get("by_level") or {}).items()
    )

    legend = "\n".join(
        f'  <span style="background:{fill};color:{font};padding:2px 8px;'
        f'border-radius:3px;margin:2px;display:inline-block">{system.value}</span>'
        for system, (fill, font) in _SYSTEM_COLOUR.items()
    )

    return f"""<div style="font-family:Helvetica,sans-serif;font-size:13px">
<h3 style="margin-bottom:4px">Lineage Graph Summary</h3>
<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;margin-bottom:8px">
{table_rows}
</table>
<h4 style="margin-bottom:4px">Edges by confidence level</h4>
<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;margin-bottom:8px">
{level_rows}
</table>
<h4 style="margin-bottom:4px">System legend</h4>
<div>{legend}</div>
</div>"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_id(node_id: str) -> str:
    """Convert a UUID to a valid Mermaid/DOT identifier."""
    return "n" + node_id.replace("-", "_")


def _display_label(node: LineageNode) -> str:
    """Human-readable node label: name (strips adf:// prefix), plus system."""
    name = node.name
    if node.full_path.startswith(_ADF_NS):
        name = f"{name}\n⚠ unresolved"
    return f"{name}\n[{node.system.value}]"


def _edge_label(edge: LineageEdge, is_gap: bool) -> str:
    """Short edge label: transformation (truncated) + confidence %."""
    parts = []
    if edge.transformation:
        t = edge.transformation
        parts.append(t[:40] + "…" if len(t) > 40 else t)
    parts.append(f"{edge.confidence:.0%}")
    if is_gap:
        parts.append("⚠ gap")
    return " | ".join(parts)


def _mermaid_arrow(edge: LineageEdge) -> str:
    """Return the Mermaid arrow syntax appropriate for the confidence level."""
    match edge.confidence_level:
        case ConfidenceLevel.CONFIRMED | ConfidenceLevel.HIGH:
            return "--"
        case ConfidenceLevel.MEDIUM:
            return "-."
        case _:
            return "~~"   # wavy for low-confidence / needs review


def _mermaid_link_style(index: int, edge: LineageEdge, is_gap: bool) -> str:
    colour = "#FF3333" if is_gap else _EDGE_COLOUR[edge.confidence_level]
    width = "3px" if is_gap or edge.confidence_level == ConfidenceLevel.CONFIRMED else "2px"
    return f"    linkStyle {index} stroke:{colour},stroke-width:{width}"


def _dot_edge_style(edge: LineageEdge, is_gap: bool) -> str:
    if is_gap:
        return 'style=bold penwidth=2'
    match edge.confidence_level:
        case ConfidenceLevel.CONFIRMED:
            return 'style=bold penwidth=2'
        case ConfidenceLevel.HIGH:
            return 'style=solid'
        case ConfidenceLevel.MEDIUM:
            return 'style=dashed'
        case _:
            return 'style=dotted penwidth=2'
