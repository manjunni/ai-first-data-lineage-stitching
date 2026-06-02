"""
Tests for the FastAPI layer.
The LangGraph agent is mocked throughout — we test routing, validation,
serialisation, and error handling, not the agent logic itself.
"""
import io
import json
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.graph.models import (
    LineageGraph, LineageNode, LineageEdge,
    LineageFragment, SystemType, ConfidenceLevel, Gap,
)
from src.graph.stitcher import stitch_fragments
from src.catalog.publisher import PublishResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph() -> LineageGraph:
    f = LineageFragment(artifact_name="t", artifact_type="t")
    raw = LineageNode(name="raw", system=SystemType.ADLS,       full_path="adls://raw")
    silver = LineageNode(name="silver", system=SystemType.DATABRICKS, full_path="silver.orders")
    f.nodes.extend([raw, silver])
    f.edges.append(LineageEdge(
        source_id=raw.id, target_id=silver.id,
        confidence=0.90, confidence_level=ConfidenceLevel.HIGH,
        transformation="filter+cast", evidence="parsed via sqlglot",
    ))
    return stitch_fragments([f])


def _make_state(run_id: str, status: str = "published", graph: LineageGraph = None,
                gaps=None, questions=None, scores=None) -> dict:
    g = graph or _make_graph()
    return {
        "run_id": run_id,
        "status": status,
        "iteration": 0,
        "lineage_graph": g,
        "gaps": gaps or [],
        "sme_questions": questions or [],
        "sme_responses": [],
        "confidence_scores": scores or {},
        "visuals": {
            "mermaid": "flowchart LR\n    nA --> nB",
            "graphviz": 'digraph lineage { nA -> nB; }',
        },
        "audit_trail": [{"step": "ingest", "ts": "2026-01-01T00:00:00Z"}],
        "target_catalog": "dry_run",
        "catalog_published": False,
    }


@pytest.fixture
def mock_agent():
    """
    Patch _get_agent() so every call returns a configured agent mock.
    `patch(..., return_value=agent_mock)` makes _get_agent() return agent_mock.
    """
    run_id = str(uuid.uuid4())
    state_values = _make_state(run_id)

    agent_mock = MagicMock()
    agent_mock.invoke.return_value = None
    agent_mock.get_state.return_value = MagicMock(values=state_values)
    agent_mock.get_state_history.return_value = []

    with patch("src.api.main._get_agent", return_value=agent_mock):
        yield agent_mock, run_id, state_values


@pytest.fixture
def client(mock_agent):
    from src.api.main import app
    return TestClient(app), mock_agent


# ---------------------------------------------------------------------------
# Helper to build a minimal file upload
# ---------------------------------------------------------------------------

def _sql_file(content: str = "INSERT INTO silver.a SELECT * FROM bronze.b") -> tuple:
    return ("files", ("test.sql", io.BytesIO(content.encode()), "text/plain"))


# ===========================================================================
# Health
# ===========================================================================

def test_health_returns_ok():
    from src.api.main import app
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "catalog_config" in data


# ===========================================================================
# POST /runs — start a run
# ===========================================================================

class TestStartRun:

    def test_returns_201_with_run_id(self, client):
        c, (mock, run_id, state) = client
        resp = c.post("/runs", files=[_sql_file()], data={"catalog": "dry_run"})
        assert resp.status_code == 201
        assert "run_id" in resp.json()

    def test_agent_invoked_once(self, client):
        c, (mock, run_id, state) = client
        c.post("/runs", files=[_sql_file()])
        mock.invoke.assert_called_once()

    def test_initial_state_contains_artifacts(self, client):
        c, (mock, run_id, state) = client
        c.post("/runs", files=[_sql_file("SELECT 1")])
        call_args = mock.invoke.call_args[0]
        initial_state = call_args[0]
        assert len(initial_state["artifacts"]) == 1
        assert initial_state["artifacts"][0]["filename"] == "test.sql"

    def test_catalog_param_in_initial_state(self, client):
        c, (mock, run_id, state) = client
        c.post("/runs", files=[_sql_file()], data={"catalog": "alation"})
        initial_state = mock.invoke.call_args[0][0]
        assert initial_state["target_catalog"] == "alation"

    def test_artifact_type_hint_passed(self, client):
        c, (mock, run_id, state) = client
        c.post("/runs", files=[_sql_file()], data={"artifact_types": "sql"})
        initial_state = mock.invoke.call_args[0][0]
        assert initial_state["artifacts"][0]["artifact_type"] == "sql"

    def test_multiple_files_uploaded(self, client):
        c, (mock, run_id, state) = client
        c.post("/runs", files=[_sql_file(), _sql_file("SELECT 2")])
        initial_state = mock.invoke.call_args[0][0]
        assert len(initial_state["artifacts"]) == 2

    def test_response_has_status(self, client):
        c, (mock, run_id, state) = client
        resp = c.post("/runs", files=[_sql_file()])
        assert "status" in resp.json()


# ===========================================================================
# GET /runs/{run_id}
# ===========================================================================

class TestGetRun:

    def test_returns_run_summary(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert "status" in data
        assert "node_count" in data
        assert "edge_count" in data

    def test_404_for_missing_run(self, client):
        c, (mock, run_id, state) = client
        mock.get_state.return_value = MagicMock(values=None)
        resp = c.get(f"/runs/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_gap_counts_correct(self, client):
        c, (mock, run_id, state) = client
        gap = Gap(gap_type="missing_source", description="test", resolved=False)
        state["gaps"] = [gap]
        resp = c.get(f"/runs/{run_id}")
        data = resp.json()
        assert data["gaps_open"] == 1
        assert data["gaps_total"] == 1


# ===========================================================================
# GET /runs/{run_id}/lineage
# ===========================================================================

class TestGetLineage:

    def test_returns_nodes_and_edges(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/lineage")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert data["node_count"] == len(state["lineage_graph"].nodes)

    def test_node_has_required_fields(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/lineage")
        node = resp.json()["nodes"][0]
        for field in ("id", "name", "system", "full_path"):
            assert field in node

    def test_edge_has_confidence(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/lineage")
        edge = resp.json()["edges"][0]
        assert "confidence" in edge
        assert "confidence_level" in edge

    def test_404_when_no_graph(self, client):
        c, (mock, run_id, state) = client
        state["lineage_graph"] = None
        resp = c.get(f"/runs/{run_id}/lineage")
        assert resp.status_code == 404


# ===========================================================================
# GET /runs/{run_id}/gaps
# ===========================================================================

class TestGetGaps:

    def _gaps(self):
        return [
            Gap(gap_type="missing_source", description="no upstream", resolved=False),
            Gap(gap_type="low_confidence_edge", description="edge uncertain", resolved=True),
        ]

    def test_returns_all_gaps(self, client):
        c, (mock, run_id, state) = client
        state["gaps"] = self._gaps()
        resp = c.get(f"/runs/{run_id}/gaps")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["open"] == 1
        assert data["resolved"] == 1

    def test_filter_open_gaps(self, client):
        c, (mock, run_id, state) = client
        state["gaps"] = self._gaps()
        resp = c.get(f"/runs/{run_id}/gaps?resolved=false")
        data = resp.json()
        assert all(not g["resolved"] for g in data["gaps"])

    def test_filter_resolved_gaps(self, client):
        c, (mock, run_id, state) = client
        state["gaps"] = self._gaps()
        resp = c.get(f"/runs/{run_id}/gaps?resolved=true")
        data = resp.json()
        assert all(g["resolved"] for g in data["gaps"])

    def test_gap_has_required_fields(self, client):
        c, (mock, run_id, state) = client
        state["gaps"] = [Gap(gap_type="missing_source", description="d")]
        resp = c.get(f"/runs/{run_id}/gaps")
        gap = resp.json()["gaps"][0]
        for field in ("id", "gap_type", "description", "resolved"):
            assert field in gap


# ===========================================================================
# GET /runs/{run_id}/questions
# ===========================================================================

class TestGetQuestions:

    def _questions(self):
        return [
            {"gap_id": str(uuid.uuid4()), "gap_type": "missing_source",
             "question": "Where does it come from?", "sme_email": "sme@corp.com",
             "answered": False, "send_status": "sent"},
            {"gap_id": str(uuid.uuid4()), "gap_type": "unknown_system",
             "question": "Which system?", "sme_email": None,
             "answered": True, "send_status": "sent"},
        ]

    def test_returns_questions(self, client):
        c, (mock, run_id, state) = client
        state["sme_questions"] = self._questions()
        resp = c.get(f"/runs/{run_id}/questions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["pending"] == 1

    def test_filter_pending(self, client):
        c, (mock, run_id, state) = client
        state["sme_questions"] = self._questions()
        resp = c.get(f"/runs/{run_id}/questions?answered=false")
        data = resp.json()
        assert all(not q["answered"] for q in data["questions"])


# ===========================================================================
# POST /runs/{run_id}/responses
# ===========================================================================

class TestSubmitResponses:

    def test_resumes_paused_run(self, client):
        c, (mock, run_id, state) = client
        state["status"] = "emails_sent"
        mock.get_state.return_value = MagicMock(values=state)

        resp = c.post(
            f"/runs/{run_id}/responses",
            json={"responses": [{"gap_id": "abc", "raw_answer": "ADLS"}]},
        )
        assert resp.status_code == 200
        mock.invoke.assert_called_once()

    def test_409_when_not_waiting(self, client):
        c, (mock, run_id, state) = client
        state["status"] = "published"
        resp = c.post(
            f"/runs/{run_id}/responses",
            json={"responses": []},
        )
        assert resp.status_code == 409

    def test_404_for_missing_run(self, client):
        c, (mock, run_id, state) = client
        mock.get_state.return_value = MagicMock(values=None)
        resp = c.post(f"/runs/{uuid.uuid4()}/responses", json={"responses": []})
        assert resp.status_code == 404


# ===========================================================================
# POST /webhooks/email-reply
# ===========================================================================

class TestEmailWebhook:

    def test_accepts_structured_reply(self, client):
        c, (mock, run_id, state) = client
        state["status"] = "emails_sent"
        gap_id = str(uuid.uuid4())
        payload = {
            "from": "sme@corp.com",
            "subject": f"Re: [Lineage Agent] Run {run_id} — Input needed",
            "body": f"[REF:{gap_id}]: The source is ADLS",
        }
        resp = c.post("/webhooks/email-reply", json=payload)
        assert resp.status_code == 202
        assert resp.json()["accepted"] is True

    def test_returns_202_even_when_run_not_found(self, client):
        c, (mock, run_id, state) = client
        mock.get_state.return_value = MagicMock(values=None)
        payload = {
            "from": "sme@corp.com",
            "subject": f"Re: [Lineage Agent] Run {uuid.uuid4()} — Input needed",
            "body": "Some answer",
        }
        resp = c.post("/webhooks/email-reply", json=payload)
        assert resp.status_code == 202

    def test_no_run_id_in_email_returns_accepted(self, client):
        c, (mock, run_id, state) = client
        payload = {"from": "sme@corp.com", "subject": "Hi there", "body": "No reference"}
        resp = c.post("/webhooks/email-reply", json=payload)
        assert resp.status_code == 202
        assert resp.json()["matched_run"] is None


# ===========================================================================
# GET /runs/{run_id}/visuals
# ===========================================================================

class TestGetVisuals:

    def test_returns_stored_visuals(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/visuals")
        assert resp.status_code == 200
        data = resp.json()
        assert "mermaid" in data
        assert "graphviz" in data

    def test_regenerates_when_not_stored(self, client):
        c, (mock, run_id, state) = client
        state["visuals"] = None
        resp = c.get(f"/runs/{run_id}/visuals")
        assert resp.status_code == 200
        assert "mermaid" in resp.json()

    def test_404_when_no_graph_and_no_visuals(self, client):
        c, (mock, run_id, state) = client
        state["visuals"] = None
        state["lineage_graph"] = None
        resp = c.get(f"/runs/{run_id}/visuals")
        assert resp.status_code == 404

    def test_summary_html_included(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/visuals")
        data = resp.json()
        assert "summary_html" in data
        assert data["summary_html"] is not None


# ===========================================================================
# GET /runs/{run_id}/confidence
# ===========================================================================

class TestGetConfidence:

    def test_returns_confidence_summary(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/confidence")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_edges" in data
        assert "by_level" in data

    def test_404_when_no_graph(self, client):
        c, (mock, run_id, state) = client
        state["lineage_graph"] = None
        resp = c.get(f"/runs/{run_id}/confidence")
        assert resp.status_code == 404


# ===========================================================================
# GET /runs/{run_id}/audit
# ===========================================================================

def test_audit_trail_returned(client):
    c, (mock, run_id, state) = client
    resp = c.get(f"/runs/{run_id}/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert "audit_trail" in data
    assert isinstance(data["audit_trail"], list)


# ===========================================================================
# Catalog publish
# ===========================================================================

class TestPublishEndpoints:

    def test_preview_returns_dry_run_result(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/publish-preview?catalog=alation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert data["success"] is True
        assert "payload_summary" in data

    def test_preview_collibra_format(self, client):
        c, (mock, run_id, state) = client
        resp = c.get(f"/runs/{run_id}/publish-preview?catalog=collibra")
        data = resp.json()
        assert data["payload_summary"]["format"] == "collibra"

    def test_publish_endpoint_calls_publisher(self, client):
        c, (mock, run_id, state) = client
        state["status"] = "finalized"

        with patch("src.api.main.get_publisher") as mock_get_pub:
            mock_pub = MagicMock()
            mock_pub.publish.return_value = PublishResult(
                status="dry_run", catalog="dry_run:alation", run_id=run_id,
                nodes_published=2, edges_published=1,
            )
            mock_get_pub.return_value = mock_pub
            resp = c.post(f"/runs/{run_id}/publish", json={"catalog": "dry_run"})

        assert resp.status_code == 200
        mock_pub.publish.assert_called_once()

    def test_publish_409_when_not_finalized(self, client):
        c, (mock, run_id, state) = client
        state["status"] = "emails_sent"
        resp = c.post(f"/runs/{run_id}/publish", json={"catalog": "dry_run"})
        assert resp.status_code == 409

    def test_preview_404_when_no_graph(self, client):
        c, (mock, run_id, state) = client
        state["lineage_graph"] = None
        resp = c.get(f"/runs/{run_id}/publish-preview")
        assert resp.status_code == 404

    def test_publish_result_has_required_fields(self, client):
        c, (mock, run_id, state) = client
        state["status"] = "finalized"
        resp = c.get(f"/runs/{run_id}/publish-preview")
        data = resp.json()
        for field in ("catalog", "status", "success", "nodes_published", "edges_published", "errors"):
            assert field in data
