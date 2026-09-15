"""
Data Lineage Stitching Agent — Streamlit UI

Run:  streamlit run src/ui/app.py
      (set API_BASE_URL env var or enter the URL in the sidebar)
"""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from src.ui.api_client import LineageApiClient, ApiError
from src.ui.components import (
    status_chip, render_mermaid, render_graphviz, render_system_legend,
    render_workflow_progress, section_header, metric_row,
    empty_state, confidence_badge, gap_type_badge, PAGE_DESCRIPTIONS,
)
from src.sme.email_handler import parse_responses_from_text
from src.sme.file_handlers import FileHandlerError, extract_text_from_file

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Data Lineage Agent",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ---------------------------------------------------------------------------
# RAISE / Capgemini theme
# ---------------------------------------------------------------------------

def _apply_raise_styles() -> None:
    st.markdown("""
    <style>
        /* ── Hide default Streamlit chrome ── */
        #MainMenu, footer,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stHeader"],
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
        }

        /* ── Collapse the top gap left by the hidden header ── */
        .block-container {
            padding-top: 0.5rem !important;
        }

        /* ── Global background ── */
        .stApp, div[data-testid="stAppViewContainer"] {
            background-color: #2d1e5f !important;
        }

        /* ── Main content area ── */
        .block-container {
            background-color: #2d1e5f !important;
            padding-top: 1rem !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background-color: #1e123a !important;
            border-right: 1px solid #3d2b70 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #e8e0ff !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #e8e0ff !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #4a3587 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #5e44a8 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #3d2b70 !important;
        }
        section[data-testid="stSidebar"] .stExpander {
            background-color: #3d2b70 !important;
            border: 1px solid #4a3587 !important;
            border-radius: 4px !important;
        }

        /* ── Text colours ── */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
            color: #e8e0ff !important;
        }
        .stCaption, small { color: #b0a8d0 !important; }

        /* ── Metric cards ── */
        div[data-testid="stMetric"] {
            background-color: #1e123a !important;
            border: 1px solid #3d2b70 !important;
            border-radius: 6px !important;
            padding: 12px !important;
        }
        div[data-testid="stMetric"] label { color: #b0a8d0 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 2rem !important;
        }

        /* ── Primary buttons ── */
        .stButton > button[kind="primary"] {
            background-color: #1e88e5 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #0d47a1 !important;
        }
        .stButton > button {
            background-color: #3d2b70 !important;
            color: #e8e0ff !important;
            border: 1px solid #4a3587 !important;
            border-radius: 4px !important;
        }
        .stButton > button:hover {
            background-color: #4a3587 !important;
        }

        /* ── Expanders ── */
        div[data-testid="stExpander"] {
            background-color: #1e123a !important;
            border: 1px solid #3d2b70 !important;
            border-radius: 6px !important;
        }
        div[data-testid="stExpander"] summary {
            color: #e8e0ff !important;
        }

        /* ── DataFrames / tables ── */
        .stDataFrame { background-color: #1e123a !important; }
        .stDataFrame th {
            background-color: #3d2b70 !important;
            color: white !important;
        }
        .stDataFrame td { color: #e8e0ff !important; }

        /* ── Tabs ── */
        div[data-testid="stTabs"] button {
            color: #b0a8d0 !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #ffffff !important;
            border-bottom-color: #1e88e5 !important;
        }

        /* ── Inputs ── */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            background-color: #ffffff !important;
            color: #333333 !important;
            border-color: #4a3587 !important;
            border-radius: 4px !important;
        }

        /* ── Code blocks ── */
        code, pre {
            background-color: #1e123a !important;
            color: #c0b8e8 !important;
            border: 1px solid #3d2b70 !important;
            border-radius: 4px !important;
        }

        /* ── Dividers ── */
        hr { border-color: #3d2b70 !important; }

        /* ── Alerts ── */
        div[data-testid="stAlert"] {
            border-radius: 4px !important;
        }

        /* ── Spinner ── */
        div[data-testid="stSpinner"] { color: #1e88e5 !important; }
    </style>
    """, unsafe_allow_html=True)


_apply_raise_styles()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "run_id":          None,
    "page":            "Dashboard",
    "api_url":         os.getenv("API_BASE_URL", "http://localhost:8000"),
    "last_error":      None,
    "last_run_result": None,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# API client (created fresh each run to pick up URL changes)
# ---------------------------------------------------------------------------

def client() -> LineageApiClient:
    return LineageApiClient(st.session_state.api_url)


def _api_call(fn, *args, **kwargs):
    """Run an API call and surface errors cleanly."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        if exc.status_code == 404:
            return None
        st.error(f"API error {exc.status_code}: {exc.detail[:300]}")
        return None
    except Exception as exc:
        st.error(f"Connection error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.image("assets/capgemini-logo.png", width=150)
        st.markdown("""
        <div style="padding-top:8px;">
            <div class="app-name">Data Lineage Stitching Agent<span style="font-size: 16px; opacity: 0.7; font-weight: normal; vertical-align: super;"></span></div>
            
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # API connection
        with st.expander("⚙️ API Settings", expanded=False):
            url = st.text_input("Backend URL", value=st.session_state.api_url,
                                key="api_url_input")
            if url != st.session_state.api_url:
                st.session_state.api_url = url
                st.rerun()
            health = _api_call(client().health)
            if health:
                st.success("Connected ✓")
                catalog_cfg = health.get("catalog_config", {})
                alation_ready  = catalog_cfg.get("alation_ready", False)
                collibra_ready = catalog_cfg.get("collibra_ready", False)
                st.caption("✅ Alation configured" if alation_ready else "⚠️ Alation: not configured")
                st.caption("✅ Collibra configured" if collibra_ready else "⚠️ Collibra: not configured")
            else:
                st.error("Cannot reach API")

        st.divider()

        # Navigation
        pages = [
            "Dashboard",
            "New Run",
            "Lineage Graph",
            "Gaps & Questions",
            "Submit Responses",
            "Confidence",
            "Publish",
            "Audit Trail",
        ]
        selected = st.radio("Navigation", pages,
                            index=pages.index(st.session_state.page))
        st.caption(PAGE_DESCRIPTIONS[selected])
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()

        # Run selector
        if st.session_state.run_id:
            st.divider()
            st.caption("Active run:")
            st.code(st.session_state.run_id[:18] + "...", language=None)
            run = _api_call(client().get_run, st.session_state.run_id)
            if run:
                status_chip(run.get("status", "unknown"))
            if st.button("Clear active run", use_container_width=True):
                st.session_state.run_id = None
                st.session_state.last_run_result = None
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard() -> None:
    section_header("📊 Dashboard", "All lineage runs")

    col_refresh, col_new, *_ = st.columns([1, 1, 4])
    if col_refresh.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    if col_new.button("➕ New Run", use_container_width=True, type="primary"):
        st.session_state.page = "New Run"
        st.rerun()

    runs = _api_call(client().list_runs) or []

    if not runs:
        empty_state("📭", "No runs yet", "Click 'New Run' to upload artifacts and start.")
        return

    # Build display table
    rows = []
    for r in runs:
        rows.append({
            "Run ID":      r["run_id"][:18] + "...",
            "Status":      r["status"],
            "Nodes":       r["node_count"],
            "Edges":       r["edge_count"],
            "Open Gaps":   r["gaps_open"],
            "_run_id":     r["run_id"],
        })

    df = pd.DataFrame(rows)
    display_df = df.drop(columns=["_run_id"])

    # Colour the status column
    def highlight_status(val):
        colours = {
            "published": "background-color:#e8f5e9",
            "emails_sent": "background-color:#f3e5f5",
            "error": "background-color:#ffebee",
            "publish_failed": "background-color:#ffebee",
        }
        return colours.get(val, "")

    st.dataframe(
        display_df.style.map(highlight_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    # Row-level actions
    st.subheader("Select a run to work with")
    run_options = {f"{r['run_id'][:18]}… ({r['status']})": r["run_id"] for r in runs}
    chosen_label = st.selectbox("Run", list(run_options.keys()), label_visibility="collapsed")
    chosen_id = run_options[chosen_label]

    action_cols = st.columns(4)
    if action_cols[0].button("🔍 Open", use_container_width=True, type="primary"):
        st.session_state.run_id = chosen_id
        st.session_state.page = "Lineage Graph"
        st.rerun()
    if action_cols[1].button("⚠️ View Gaps", use_container_width=True):
        st.session_state.run_id = chosen_id
        st.session_state.page = "Gaps & Questions"
        st.rerun()
    if action_cols[2].button("📧 Respond", use_container_width=True):
        st.session_state.run_id = chosen_id
        st.session_state.page = "Submit Responses"
        st.rerun()
    if action_cols[3].button("📤 Publish", use_container_width=True):
        st.session_state.run_id = chosen_id
        st.session_state.page = "Publish"
        st.rerun()


# ---------------------------------------------------------------------------
# Page: New Run
# ---------------------------------------------------------------------------

def page_new_run() -> None:
    section_header("🚀 New Run", "Upload artifacts to start lineage extraction")

    with st.form("new_run_form"):
        st.subheader("1. Upload artifacts")
        uploaded = st.file_uploader(
            "SQL scripts, Databricks notebooks, DLT pipelines, ADF pipeline JSON",
            accept_multiple_files=True,
            type=["sql", "ipynb", "py", "json"],
        )

        st.subheader("2. Configuration")
        col_catalog, col_types = st.columns(2)
        catalog = col_catalog.selectbox(
            "Target catalog",
            ["dry_run", "alation", "collibra"],
            index=0,
            help="dry_run: serialise payload without publishing. Use alation/collibra once credentials are configured.",
        )
        artifact_types_hint = col_types.text_input(
            "Artifact type hints (optional)",
            placeholder="sql,notebook,dlt,adf  (one per file, in order)",
            help="Leave blank for auto-detection. Use this to disambiguate e.g. a .json that is an ADF pipeline.",
        )

        submitted = st.form_submit_button("▶️ Start Run", type="primary", use_container_width=True)

    if submitted:
        if not uploaded:
            st.error("Please upload at least one artifact file.")
            return

        with st.spinner("Starting run — parsing artifacts and building lineage graph…"):
            files = [
                ("files", (f.name, io.BytesIO(f.read()), "application/octet-stream"))
                for f in uploaded
            ]
            result = _api_call(
                client().start_run,
                files=files,
                catalog=catalog,
                artifact_types=artifact_types_hint,
            )

        if result:
            run_id = result["run_id"]
            st.session_state.run_id = run_id
            st.session_state.last_run_result = result
            st.success(f"Run started: `{run_id}`")
        else:
            st.error("Failed to start run. Check that the API is running.")

    if st.session_state.last_run_result and st.session_state.run_id:
        _show_run_result(st.session_state.last_run_result)


def _show_run_result(run: dict) -> None:
    """Show a summary card after starting or refreshing a run."""
    status = run.get("status", "unknown")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", run.get("node_count", 0))
    col2.metric("Edges", run.get("edge_count", 0))
    col3.metric("Open Gaps", run.get("gaps_open", 0))
    col4.metric("Iteration", run.get("iteration", 0))

    render_workflow_progress(status)

    if status == "emails_sent":
        st.warning(
            "⏸️ Waiting for SME responses. Questions have been emailed. "
            "Submit replies via the **Submit Responses** page."
        )
        if st.button("➡️ Go to Submit Responses"):
            st.session_state.page = "Submit Responses"
            st.rerun()
    elif status in ("finalized", "visuals_ready", "published"):
        st.success("✅ Lineage extraction complete!")
        if st.button("🗺️ View Lineage Graph"):
            st.session_state.page = "Lineage Graph"
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Lineage Graph
# ---------------------------------------------------------------------------

def page_lineage_graph() -> None:
    section_header("🕸️ Lineage Graph", "Visual cross-system lineage map")

    run_id = _require_run()
    if not run_id:
        return

    run = _api_call(client().get_run, run_id)
    if run:
        render_workflow_progress(run.get("status", ""))
        metric_row([
            ("Nodes",      str(run.get("node_count", 0)), None),
            ("Edges",      str(run.get("edge_count", 0)), None),
            ("Open Gaps",  str(run.get("gaps_open", 0)),  None),
            ("Iteration",  str(run.get("iteration", 0)),  None),
        ])

    st.subheader("System legend")
    render_system_legend()
    st.markdown("")

    # Diagram
    tab_mermaid, tab_dot, tab_nodes, tab_edges = st.tabs([
        "📊 Mermaid Diagram", "🔵 Graphviz DOT", "🗂️ Nodes Table", "↔️ Edges Table"
    ])

    with tab_mermaid:
        visuals = _api_call(client().get_visuals, run_id)
        if visuals:
            height = st.slider("Diagram height", 300, 1200, 500, 50, key="mermaid_height")
            render_mermaid(visuals.get("mermaid", ""), height=height)

            with st.expander("📋 Raw Mermaid source"):
                st.code(visuals.get("mermaid", ""), language="text")
        else:
            empty_state("⌛", "Diagram not yet generated",
                        "The run must reach the 'visuals_ready' stage.")

    with tab_dot:
        if visuals:
            dot = visuals.get("graphviz", "")
            height_dot = st.slider("Diagram height", 300, 1200, 600, 50, key="dot_height")
            render_graphviz(dot, height=height_dot)
            with st.expander("📋 Raw DOT source"):
                st.code(dot, language="dot")
        else:
            empty_state("⌛", "Not yet available")

    with tab_nodes:
        lineage = _api_call(client().get_lineage, run_id)
        if lineage and lineage.get("nodes"):
            def _display_path(n: dict) -> str:
                # For adf:// placeholder nodes, the actual storage path is
                # in the description field — show that instead.
                fp = n["full_path"]
                if fp.startswith("adf://") and n.get("description"):
                    return n["description"]
                return fp

            df = pd.DataFrame([
                {
                    "Name":      n["name"],
                    "System":    n["system"],
                    "Full Path": _display_path(n),
                    "Owner":     n.get("owner") or "—",
                }
                for n in lineage["nodes"]
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            empty_state("🗂️", "No nodes yet")

    with tab_edges:
        if lineage and lineage.get("edges"):
            df = pd.DataFrame([
                {
                    "Source":        _node_name(lineage, e["source_id"]),
                    "→ Target":      _node_name(lineage, e["target_id"]),
                    "Transformation":e.get("transformation") or "—",
                    "Confidence":    f'{e["confidence"]:.0%}',
                    "Level":         e.get("confidence_level", "—"),
                    "Boundary":      e.get("system_boundary") or "—",
                    "SME Approved":  e.get("sme_approved_by") or "—",
                }
                for e in lineage["edges"]
            ])

            def colour_level(val):
                colours = {"confirmed": "#e3f2fd", "high": "#e8f5e9",
                           "medium": "#fff8e1", "low": "#ffebee"}
                return f"background-color:{colours.get(val, '')}"

            st.dataframe(
                df.style.map(colour_level, subset=["Level"]),
                use_container_width=True, hide_index=True,
            )
        else:
            empty_state("↔️", "No edges yet")


def _node_name(lineage: dict, node_id: str) -> str:
    for n in lineage.get("nodes", []):
        if n["id"] == node_id:
            return n["name"]
    return node_id[:12]


# ---------------------------------------------------------------------------
# Page: Gaps & Questions
# ---------------------------------------------------------------------------

def page_gaps_and_questions() -> None:
    section_header("⚠️ Gaps & Questions", "Open lineage gaps and generated SME questions")

    run_id = _require_run()
    if not run_id:
        return

    tab_gaps, tab_questions = st.tabs(["🔍 Gaps", "📧 SME Questions"])

    with tab_gaps:
        col_filter, _ = st.columns([1, 3])
        gap_filter = col_filter.selectbox(
            "Show", ["All", "Open only", "Resolved only"], key="gap_filter"
        )
        resolved_param = None if gap_filter == "All" else gap_filter == "Resolved only"

        gaps_data = _api_call(client().get_gaps, run_id, resolved=resolved_param)
        if not gaps_data:
            return

        metric_row([
            ("Total Gaps",    str(gaps_data.get("total", 0)),    None),
            ("Open",          str(gaps_data.get("open", 0)),     None),
            ("Resolved",      str(gaps_data.get("resolved", 0)), None),
        ])

        gaps = gaps_data.get("gaps", [])
        if not gaps:
            empty_state("✅", "No gaps found", "The lineage graph is complete.")
            return

        for gap in gaps:
            icon = "✅" if gap["resolved"] else "⚠️"
            with st.expander(
                f"{icon} {gap_type_badge(gap['gap_type'])} — {gap['description'][:80]}",
                expanded=not gap["resolved"],
            ):
                st.markdown(f"**Description:** {gap['description']}")
                if gap.get("node_id"):
                    st.caption(f"Node ID: `{gap['node_id']}`")
                if gap.get("edge_id"):
                    st.caption(f"Edge ID: `{gap['edge_id']}`")
                if gap.get("sme_email"):
                    st.caption(f"📧 Assigned SME: {gap['sme_email']}")
                st.caption(f"Gap ID: `{gap['id']}`")

    with tab_questions:
        col_q_filter, _ = st.columns([1, 3])
        q_filter = col_q_filter.selectbox(
            "Show", ["All", "Pending only", "Answered only"], key="q_filter"
        )
        answered_param = None if q_filter == "All" else q_filter == "Answered only"

        q_data = _api_call(client().get_questions, run_id, answered=answered_param)
        if not q_data:
            return

        metric_row([
            ("Total Questions", str(q_data.get("total", 0)),   None),
            ("Pending",         str(q_data.get("pending", 0)), None),
        ])

        questions = q_data.get("questions", [])
        if not questions:
            empty_state("📭", "No questions generated yet",
                        "Questions are generated when gaps are detected.")
            return

        for q in questions:
            status_icon = "✅" if q["answered"] else "⏳"
            label = f"{status_icon} [{q['gap_type']}] {q['question'][:80]}"
            with st.expander(label, expanded=not q["answered"]):
                st.markdown(f"**Question:**\n\n> {q['question']}")
                cols = st.columns(3)
                cols[0].caption(f"Gap ID: `{q['gap_id']}`")
                cols[1].caption(f"SME: {q.get('sme_email') or 'not assigned'}")
                cols[2].caption(f"Send status: {q.get('send_status') or '—'}")

                if not q["answered"]:
                    # Copy-to-clipboard helper
                    ref = q["gap_id"]
                    st.code(f"[REF:{ref}]: <your answer here>", language="text")
                    st.caption("Include the line above in your email reply.")


# ---------------------------------------------------------------------------
# Page: Submit Responses
# ---------------------------------------------------------------------------

def page_submit_responses() -> None:
    section_header("📧 Submit SME Responses",
                   "Paste email replies or upload response files to resume the lineage run")

    run_id = _require_run()
    if not run_id:
        return

    run = _api_call(client().get_run, run_id)
    if not run:
        return

    status = run.get("status", "")
    if status != "emails_sent":
        if status in ("finalized", "visuals_ready", "published"):
            st.success("✅ All gaps resolved — no responses needed.")
            if st.button("🗺️ View Lineage Graph"):
                st.session_state.page = "Lineage Graph"
                st.rerun()
        else:
            st.info(f"This run is currently in status **{status}**. "
                    "Responses can only be submitted when status is `emails_sent`.")
        return

    # Show each question with its answer input directly below it
    q_data = _api_call(client().get_questions, run_id, answered=False)
    questions = q_data.get("questions", []) if q_data else []

    if not questions:
        st.info("No pending questions found.")
        return

    st.subheader(f"📋 {len(questions)} question{'s' if len(questions) != 1 else ''} awaiting answers")
    responder_name = st.text_input(
        "Your name or email (appears as owner of resolved nodes)",
        key="responder_name",
        placeholder="e.g. Manjusha Unnikrishnan",
    )

    upload_answers = {}
    upload_errors = []

    # Each question accepts either typed text or an uploaded document. The
    # extracted document text is associated with this question's gap ID.
    for i, q in enumerate(questions):
        gap_id = q["gap_id"]
        answer_key = f"answer_{gap_id}"
        st.markdown(f"**Q{i + 1} — {gap_type_badge(q['gap_type'])}**")
        st.markdown(f"> {q['question']}")
        st.text_input(
                    "Your answer",
                    key=answer_key,
                    placeholder="Type your answer here…",
                    label_visibility="collapsed",
                )
        uploaded = st.file_uploader(
            "OR Upload a response document",
            type=["txt", "docx", "xlsx", "pdf", "json"],
            key=f"response_file_{gap_id}",
            help="Upload a document containing the answer to this question.",
        )
        if uploaded is not None:
            try:
                extracted_text = extract_text_from_file(
                    uploaded.name, uploaded.getvalue()
                ).strip()
                parsed = parse_responses_from_text(
                    extracted_text,
                    sme_email=responder_name.strip() or "analyst@ui",
                    run_id=run_id,
                )
                matching = [
                    item["raw_answer"].strip()
                    for item in parsed
                    if item.get("gap_id") == gap_id and item.get("raw_answer", "").strip()
                ]
                upload_answers[gap_id] = "\n".join(matching) or extracted_text
                if not upload_answers[gap_id]:
                    upload_errors.append(f"{uploaded.name}: no answer text found")
                else:
                    st.caption(f"📄 Uploaded answer ready: {uploaded.name}")
            except (FileHandlerError, ValueError) as exc:
                upload_errors.append(f"{uploaded.name}: {exc}")
            except Exception as exc:
                upload_errors.append(f"{uploaded.name}: {exc}")
        
        st.divider()

    col_submit, col_clear = st.columns([2, 1])
    submit = col_submit.button("📨 Submit All Answers", type="primary", use_container_width=True)
    if col_clear.button("🗑️ Clear All", use_container_width=True):
        for q in questions:
            st.session_state.pop(f"answer_{q['gap_id']}", None)
            st.session_state.pop(f"response_file_{q['gap_id']}", None)
        st.rerun()

    if submit:
        for error in upload_errors:
            st.error(error)
        if upload_errors:
            st.error("Fix the uploaded document errors before submitting.")
            return

        # Build [REF:gap_id] answers from either typed text or uploaded files.
        lines = []
        for q in questions:
            answer = st.session_state.get(f"answer_{q['gap_id']}", "").strip()
            uploaded_answer = upload_answers.get(q["gap_id"], "")
            if answer and uploaded_answer:
                answer = f"{answer}\n{uploaded_answer}"
            else:
                answer = answer or uploaded_answer
            if answer:
                lines.append(f"[REF:{q['gap_id']}]: {answer}")

        if not lines:
            st.error("Please provide at least one answer before submitting.")
            return

        reply_text = "\n".join(lines)
        from src.sme.email_handler import parse_email_responses
        sender = st.session_state.get("responder_name", "").strip() or "analyst@ui"
        raw = [{"from": sender, "subject": f"[Lineage Agent] Run {run_id} — reply",
                "body": reply_text}]
        parsed = parse_email_responses(raw)

        for p in parsed:
            p["run_id"] = run_id

        with st.spinner("Processing responses and updating the lineage graph…"):
            result = _api_call(client().submit_responses, run_id, parsed)

        if result:
            st.session_state.last_run_result = result
            # Clear answer inputs after successful submission
            for q in questions:
                st.session_state.pop(f"answer_{q['gap_id']}", None)
            st.success("✅ Responses submitted! The run is continuing.")
        else:
            st.error("Failed to submit responses.")

    if st.session_state.last_run_result and st.session_state.run_id:
        _show_run_result(st.session_state.last_run_result)


# ---------------------------------------------------------------------------
# Page: Confidence
# ---------------------------------------------------------------------------

def page_confidence() -> None:
    section_header("📊 Confidence Scores",
                   "How confident is the extracted lineage?")

    run_id = _require_run()
    if not run_id:
        return

    data = _api_call(client().get_confidence, run_id)
    if not data:
        empty_state("📊", "Confidence data not yet available",
                    "The run must be finalized first.")
        return

    avg = data.get("avg_confidence")
    metric_row([
        ("Total Edges",      str(data.get("total_edges", 0)),       None),
        ("Avg Confidence",   f"{avg:.0%}" if avg else "—",          None),
        ("SME Confirmed",    str(data.get("sme_confirmed", 0)),      None),
        ("Cross-system",     str(data.get("cross_system_edges", 0)), None),
    ])

    st.divider()

    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        st.subheader("Edges by confidence level")
        by_level = data.get("by_level", {})
        if by_level:
            chart_data = pd.DataFrame({
                "Level":  list(by_level.keys()),
                "Count":  list(by_level.values()),
            }).set_index("Level")
            st.bar_chart(chart_data, color="#4472C4")

    with col_stats:
        st.subheader("Statistics")
        if avg is not None:
            st.progress(avg, text=f"Average: {avg:.0%}")
        st.metric("Min confidence", f"{data.get('min_confidence', 0):.0%}")
        st.metric("Max confidence", f"{data.get('max_confidence', 0):.0%}")
        if data.get("unresolved_adf_edges", 0):
            st.warning(f"⚠️ {data['unresolved_adf_edges']} unresolved ADF dataset references")

    # Confidence level explanations
    with st.expander("ℹ️ What do confidence levels mean?"):
        rows = [
            ("✅ Confirmed",  ">= 95%", "SME has explicitly approved this lineage link"),
            ("🟢 High",       ">= 85%", "Parsed directly from artifact (SQL AST, Python AST, ADF JSON)"),
            ("🟡 Medium",     ">= 65%", "Inferred by LLM or parsed via regex"),
            ("🔴 Low",        "< 65%",  "Uncertain — flagged as a gap for SME review"),
        ]
        st.table(pd.DataFrame(rows, columns=["Level", "Threshold", "Meaning"]))


# ---------------------------------------------------------------------------
# Page: Publish
# ---------------------------------------------------------------------------

def page_publish() -> None:
    section_header("📤 Publish to Catalog",
                   "Preview and publish the validated lineage to Alation or Collibra")

    run_id = _require_run()
    if not run_id:
        return

    run = _api_call(client().get_run, run_id)
    if not run:
        return

    status = run.get("status", "")
    PUBLISHABLE = {"finalized", "visuals_ready", "published", "publish_failed"}

    if status not in PUBLISHABLE:
        st.warning(
            f"Run status is **{status}**. "
            "Publishing is available after the run is finalized."
        )
        render_workflow_progress(status)
        return

    # Compute catalog_ready BEFORE rendering the columns so the button's
    # disabled state reflects the current selection correctly.
    health = _api_call(client().health)
    catalog_cfg = health.get("catalog_config", {}) if health else {}

    catalog = st.selectbox(
        "Target catalog",
        ["dry_run", "alation", "collibra"],
        index=["dry_run", "alation", "collibra"].index(
            run.get("target_catalog", "dry_run")
        ),
        key="publish_catalog",
    )
    catalog_ready = catalog == "dry_run" or catalog_cfg.get(f"{catalog}_ready", False)

    col_catalog, col_action = st.columns([2, 2])

    if catalog != "dry_run" and not catalog_ready:
        st.warning(
            f"⚠️ **{catalog.capitalize()} is not configured.** "
            f"You can still preview the payload below. "
            f"To publish, set `{catalog.upper()}_BASE_URL` and credentials in `.env`."
        )

    st.divider()

    # Preview
    st.subheader("📋 Payload preview")
    st.caption("See exactly what will be sent to the catalog before committing.")

    if st.button("👁️ Preview payload", use_container_width=False):
        with st.spinner("Generating preview…"):
            preview = _api_call(client().publish_preview, run_id, catalog=catalog)
        if preview:
            _render_publish_result(preview)

    st.divider()

    # Publish
    st.subheader("🚀 Publish")
    if catalog == "dry_run":
        st.info("**dry_run** — the payload will be serialised and returned but NOT sent to any catalog. Safe to use for testing.")

    if st.button(
        f"📤 Publish to {catalog}", type="primary",
        disabled=not catalog_ready,
        help=None if catalog_ready else f"Configure {catalog.capitalize()} credentials in .env first",
    ):
        with st.spinner(f"Publishing to {catalog}…"):
            result = _api_call(client().publish, run_id, catalog=catalog)
        if result:
            if result.get("success"):
                st.success(f"✅ Published successfully to {catalog}!")
            else:
                st.error(f"Publish failed: {result.get('errors', [])}")
            _render_publish_result(result)


def _render_publish_result(result: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Status", result.get("status", "—"))
    cols[1].metric("Nodes", result.get("nodes_published", 0))
    cols[2].metric("Edges", result.get("edges_published", 0))
    cols[3].metric("Errors", len(result.get("errors", [])))

    if result.get("errors"):
        with st.expander("❌ Errors"):
            for e in result["errors"]:
                st.error(e)

    if result.get("catalog_urls"):
        with st.expander("🔗 Catalog links"):
            for url in result["catalog_urls"]:
                st.markdown(f"- [{url}]({url})")

    if result.get("payload_summary"):
        payload = result["payload_summary"]
        # Show a human-readable breakdown alongside the raw JSON
        if "lineage" in payload:
            items = payload["lineage"]
            st.subheader(f"📦 Alation payload — {len(items)} lineage item{'s' if len(items) != 1 else ''}")
            for i, item in enumerate(items):
                df_name = item.get("dataflow", {}).get("name", f"item {i+1}")
                with st.expander(f"🔗 {df_name}", expanded=i == 0):
                    st.json(item)
        elif "assets" in payload:
            assets    = payload.get("assets", [])
            relations = payload.get("relations", [])
            st.subheader(f"📦 Collibra payload — {len(assets)} assets, {len(relations)} relations")
            tab_a, tab_r = st.tabs(["Assets", "Relations"])
            with tab_a:
                st.json(assets)
            with tab_r:
                st.json(relations)
        else:
            with st.expander("📦 Full payload"):
                st.json(payload)


# ---------------------------------------------------------------------------
# Page: Audit Trail
# ---------------------------------------------------------------------------

def page_audit_trail() -> None:
    section_header("📜 Audit Trail",
                   "Full history of every action taken during this run")

    run_id = _require_run()
    if not run_id:
        return

    audit = _api_call(client().get_audit, run_id)
    if not audit:
        return

    trail = audit.get("audit_trail", [])
    if not trail:
        empty_state("📜", "Audit trail is empty")
        return

    st.caption(f"{len(trail)} audit entries")

    # Reverse so most recent is first
    for entry in reversed(trail):
        step = entry.get("step", "unknown")
        ts = entry.get("ts", "")
        details = {k: v for k, v in entry.items() if k not in ("step", "ts")}

        with st.expander(f"🔹 **{step}** — {ts[:19]}", expanded=False):
            if details:
                st.json(details)
            else:
                st.caption("No additional details")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_run() -> str | None:
    """Return the active run_id or show a prompt and return None."""
    if not st.session_state.run_id:
        st.info("No active run selected. Choose a run from the **Dashboard** or start a **New Run**.")
        col1, col2, *_ = st.columns([1, 1, 4])
        if col1.button("📊 Go to Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
        if col2.button("➕ New Run"):
            st.session_state.page = "New Run"
            st.rerun()
        return None
    return st.session_state.run_id


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PAGES = {
    "Dashboard":        page_dashboard,
    "New Run":          page_new_run,
    "Lineage Graph":    page_lineage_graph,
    "Gaps & Questions": page_gaps_and_questions,
    "Submit Responses": page_submit_responses,
    "Confidence":       page_confidence,
    "Publish":          page_publish,
    "Audit Trail":      page_audit_trail,
}


def main() -> None:
    render_sidebar()
    page_fn = _PAGES.get(st.session_state.page, page_dashboard)
    page_fn()


if __name__ == "__main__":
    main()
