from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as stc

# ---------------------------------------------------------------------------
# Status colours
# ---------------------------------------------------------------------------

_STATUS_COLOUR = {
    "started":             ("🔵", "#4472C4"),
    "ingested":            ("🔵", "#4472C4"),
    "parsed":              ("🟡", "#FFC000"),
    "stitched":            ("🟡", "#FFC000"),
    "gaps_detected":       ("🟠", "#ED7D31"),
    "questions_generated": ("🟠", "#ED7D31"),
    "emails_sent":         ("🟣", "#7030A0"),
    "responses_received":  ("🟣", "#7030A0"),
    "responses_processed": ("🟣", "#7030A0"),
    "finalized":           ("🟢", "#70AD47"),
    "visuals_ready":       ("🟢", "#70AD47"),
    "published":           ("✅", "#70AD47"),
    "publish_failed":      ("❌", "#FF0000"),
    "error":               ("❌", "#FF0000"),
}

_GAP_TYPE_ICON = {
    "missing_source":         "🔴",
    "missing_target":         "🟠",
    "unknown_system":         "❓",
    "unresolved_dataset_ref": "📎",
    "low_confidence_edge":    "⚠️",
    "contradictory_paths":    "⚡",
    "isolated_subgraph":      "🏝️",
}

_SYSTEM_COLOUR = {
    "adls":        "#4472C4",
    "databricks":  "#FF8C00",
    "sql":         "#70AD47",
    "dlt":         "#7030A0",
    "adf":         "#ED7D31",
    "bi":          "#FFC000",
    "unknown":     "#A6A6A6",
}

_CONFIDENCE_COLOUR = {
    "confirmed": "#4472C4",
    "high":      "#70AD47",
    "medium":    "#FFC000",
    "low":       "#FF0000",
}


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------

def status_badge(status: str) -> str:
    icon, colour = _STATUS_COLOUR.get(status, ("⚪", "#888888"))
    return f'{icon} <span style="color:{colour};font-weight:600">{status}</span>'


def status_chip(status: str) -> None:
    icon, colour = _STATUS_COLOUR.get(status, ("⚪", "#888888"))
    st.markdown(
        f'<span style="background:{colour}22;color:{colour};padding:3px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:600">{icon} {status}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Mermaid diagram
# ---------------------------------------------------------------------------

def render_mermaid(diagram: str, height: int = 500) -> None:
    """Render a Mermaid diagram using the CDN-loaded mermaid.js."""
    if not diagram:
        st.info("No diagram available yet.")
        return

    escaped = diagram.replace("`", "\\`")
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body {{ margin: 0; background: white; }}
    .mermaid {{ font-family: Helvetica, Arial, sans-serif; }}
  </style>
</head>
<body>
  <div class="mermaid">
{diagram}
  </div>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {{ useMaxWidth: true, htmlLabels: true }}
    }});
  </script>
</body>
</html>"""
    stc.html(html, height=height, scrolling=True)


# ---------------------------------------------------------------------------
# System legend
# ---------------------------------------------------------------------------

def render_system_legend() -> None:
    cols = st.columns(len(_SYSTEM_COLOUR))
    for col, (system, colour) in zip(cols, _SYSTEM_COLOUR.items()):
        col.markdown(
            f'<div style="background:{colour};color:white;text-align:center;'
            f'padding:4px;border-radius:4px;font-size:12px">{system.upper()}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Confidence badge
# ---------------------------------------------------------------------------

def confidence_badge(level: str, score: float) -> str:
    colour = _CONFIDENCE_COLOUR.get(level, "#888")
    return (
        f'<span style="background:{colour}22;color:{colour};padding:2px 8px;'
        f'border-radius:10px;font-size:12px">{score:.0%} ({level})</span>'
    )


# ---------------------------------------------------------------------------
# Gap type badge
# ---------------------------------------------------------------------------

def gap_type_badge(gap_type: str) -> str:
    icon = _GAP_TYPE_ICON.get(gap_type, "🔷")
    label = gap_type.replace("_", " ")
    return f"{icon} {label}"


# ---------------------------------------------------------------------------
# Progress timeline
# ---------------------------------------------------------------------------

_WORKFLOW_STEPS = [
    ("ingest_artifacts",  "📥 Ingest"),
    ("parse_artifacts",   "🔍 Parse"),
    ("stitch_graph",      "🧵 Stitch"),
    ("detect_gaps",       "🔎 Gaps"),
    ("sme_loop",          "📧 SME Loop"),
    ("score_and_finalize","📊 Score"),
    ("generate_visuals",  "🎨 Visualise"),
    ("publish_to_catalog","📤 Publish"),
]

_STATUS_TO_STEP = {
    "started":             0,
    "ingested":            1,
    "parsed":              2,
    "stitched":            3,
    "gaps_detected":       4,
    "questions_generated": 4,
    "emails_sent":         4,
    "responses_received":  4,
    "responses_processed": 4,
    "finalized":           5,
    "visuals_ready":       6,
    "published":           7,
    "publish_failed":      7,
}


def render_workflow_progress(status: str) -> None:
    current = _STATUS_TO_STEP.get(status, 0)
    cols = st.columns(len(_WORKFLOW_STEPS))
    for i, (col, (_, label)) in enumerate(zip(cols, _WORKFLOW_STEPS)):
        if i < current:
            col.markdown(
                f'<div style="text-align:center;color:#70AD47;font-size:11px">'
                f'✅<br>{label}</div>', unsafe_allow_html=True
            )
        elif i == current:
            col.markdown(
                f'<div style="text-align:center;color:#4472C4;font-weight:bold;font-size:11px">'
                f'▶️<br>{label}</div>', unsafe_allow_html=True
            )
        else:
            col.markdown(
                f'<div style="text-align:center;color:#aaa;font-size:11px">'
                f'⬜<br>{label}</div>', unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------

def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


# ---------------------------------------------------------------------------
# Metric row
# ---------------------------------------------------------------------------

def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """metrics: list of (label, value, delta_or_None)"""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def empty_state(icon: str, message: str, hint: str = "") -> None:
    hint_html = f'<div style="font-size:13px;margin-top:4px">{hint}</div>' if hint else ""
    st.markdown(
        f'<div style="text-align:center;padding:48px;color:#666">'
        f'<div style="font-size:48px">{icon}</div>'
        f'<div style="font-size:18px;margin-top:8px">{message}</div>'
        f'{hint_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
