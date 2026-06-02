"""
Tests for the SME email flow.
LLM calls are mocked throughout; only pure logic is tested directly.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.sme.question_generator import (
    generate_questions, _fill_template, _resolve_sme_email,
)
from src.sme.email_handler import (
    send_sme_questions, parse_email_responses,
    _format_plain, _format_html, _extract_ref_answers, _extract_run_id,
)
from src.sme.response_processor import (
    apply_responses_to_graph,
    _parse_system, _find_or_create_node,
)
from src.graph.models import (
    Gap, LineageGraph, LineageNode, LineageEdge,
    SystemType, ConfidenceLevel,
)
from src.graph.stitcher import stitch_fragments
from src.graph.models import LineageFragment
import uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(name: str, system: SystemType = SystemType.DATABRICKS,
          full_path: str | None = None, owner: str | None = None) -> LineageNode:
    return LineageNode(name=name, system=system, full_path=full_path or name, owner=owner)


def _edge(src: LineageNode, tgt: LineageNode,
          confidence: float = 0.60, evidence: str = "llm") -> LineageEdge:
    return LineageEdge(
        source_id=src.id, target_id=tgt.id,
        confidence=confidence, evidence=evidence,
    )


def _graph(*items) -> LineageGraph:
    f = LineageFragment(artifact_name="t", artifact_type="t")
    for item in items:
        if isinstance(item, LineageNode):
            f.nodes.append(item)
        else:
            f.edges.append(item)
    return stitch_fragments([f])


def _gap(gap_type: str, node_id: str | None = None, edge_id: str | None = None,
         description: str = "test gap") -> Gap:
    return Gap(gap_type=gap_type, node_id=node_id, edge_id=edge_id, description=description)


# ===========================================================================
# Question generator — template filling (no LLM)
# ===========================================================================

class TestTemplateFilling:

    def test_missing_source_includes_node_name(self):
        n = _node("silver_orders", system=SystemType.DATABRICKS)
        g = _graph(n)
        gap = _gap("missing_source", node_id=list(g.nodes.keys())[0])
        q, _ = _fill_template(gap, g)
        assert "silver_orders" in q
        assert "databricks" in q.lower()

    def test_missing_target_includes_node_name(self):
        n = _node("gold_summary", system=SystemType.DATABRICKS)
        g = _graph(n)
        gap = _gap("missing_target", node_id=list(g.nodes.keys())[0])
        q, _ = _fill_template(gap, g)
        assert "gold_summary" in q

    def test_unknown_system_includes_path(self):
        n = _node("mystery", system=SystemType.UNKNOWN, full_path="mystery_path")
        g = _graph(n)
        gap = _gap("unknown_system", node_id=list(g.nodes.keys())[0])
        q, _ = _fill_template(gap, g)
        assert "mystery" in q
        assert "mystery_path" in q

    def test_unresolved_dataset_ref_strips_adf_prefix(self):
        n = _node("MyDS", system=SystemType.UNKNOWN, full_path="adf://MyDS")
        g = _graph(n)
        gap = _gap("unresolved_dataset_ref", node_id=list(g.nodes.keys())[0])
        q, _ = _fill_template(gap, g)
        assert "MyDS" in q
        assert "adf://" not in q

    def test_low_confidence_edge_includes_confidence_percent(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        e = _edge(src, tgt, confidence=0.55)
        g = _graph(src, tgt, e)
        gap = _gap("low_confidence_edge", edge_id=g.edges[0].id)
        q, _ = _fill_template(gap, g)
        assert "55%" in q

    def test_low_confidence_includes_node_names(self):
        src = _node("raw_orders", full_path="raw_orders")
        tgt = _node("silver_orders", full_path="silver_orders")
        e = _edge(src, tgt, confidence=0.60)
        g = _graph(src, tgt, e)
        gap = _gap("low_confidence_edge", edge_id=g.edges[0].id)
        q, _ = _fill_template(gap, g)
        assert "raw_orders" in q
        assert "silver_orders" in q

    def test_unknown_gap_type_falls_back_to_description(self):
        g = LineageGraph()
        gap = _gap("exotic_future_gap_type", description="Some exotic gap")
        q, _ = _fill_template(gap, g)
        assert "exotic" in q.lower()


class TestSmeEmailResolution:

    def test_gap_sme_email_used_first(self):
        g = LineageGraph()
        gap = Gap(gap_type="missing_source", description="d", sme_email="direct@corp.com")
        assert _resolve_sme_email(gap, g) == "direct@corp.com"

    def test_node_owner_used_as_fallback(self):
        n = _node("t", owner="owner@corp.com")
        g = _graph(n)
        gap = _gap("missing_source", node_id=list(g.nodes.keys())[0])
        assert _resolve_sme_email(gap, g) == "owner@corp.com"

    def test_none_returned_when_no_email_and_no_default(self):
        g = LineageGraph()
        gap = _gap("missing_source")
        with patch("src.sme.question_generator.settings") as mock_settings:
            del mock_settings.default_sme_email
            result = _resolve_sme_email(gap, g)
        assert result is None


class TestGenerateQuestionsNoLlm:

    def test_returns_one_question_per_open_gap(self):
        n = _node("t", full_path="t")
        g = _graph(n)
        node_id = list(g.nodes.keys())[0]
        gaps = [
            _gap("missing_source", node_id=node_id),
            _gap("missing_target", node_id=node_id),
        ]
        questions = generate_questions(gaps, g, enrich_with_llm=False)
        assert len(questions) == 2

    def test_resolved_gaps_skipped(self):
        g = LineageGraph()
        gap = Gap(gap_type="missing_source", description="d", resolved=True)
        questions = generate_questions([gap], g, enrich_with_llm=False)
        assert len(questions) == 0

    def test_question_dict_has_required_keys(self):
        n = _node("t", full_path="t")
        g = _graph(n)
        gap = _gap("missing_source", node_id=list(g.nodes.keys())[0])
        questions = generate_questions([gap], g, enrich_with_llm=False)
        q = questions[0]
        assert "gap_id" in q
        assert "gap_type" in q
        assert "question" in q
        assert "answered" in q
        assert q["answered"] is False

    def test_gap_id_preserved(self):
        n = _node("t", full_path="t")
        g = _graph(n)
        gap = _gap("missing_source", node_id=list(g.nodes.keys())[0])
        questions = generate_questions([gap], g, enrich_with_llm=False)
        assert questions[0]["gap_id"] == gap.id


# ===========================================================================
# Email handler — formatting (pure functions)
# ===========================================================================

class TestEmailFormatting:

    def _questions(self, n: int = 2) -> list[dict]:
        return [
            {
                "gap_id": str(uuid.uuid4()),
                "gap_type": "missing_source",
                "question": f"Where does table_{i} come from?",
                "sme_email": "sme@corp.com",
                "run_id": "run-123",
                "answered": False,
            }
            for i in range(n)
        ]

    def test_plain_contains_all_gap_refs(self):
        qs = self._questions(2)
        plain = _format_plain(qs, "run-123")
        for q in qs:
            assert f"[REF:{q['gap_id']}]" in plain

    def test_plain_contains_run_id(self):
        qs = self._questions()
        plain = _format_plain(qs, "run-abc")
        assert "run-abc" in plain

    def test_html_contains_all_gap_refs(self):
        qs = self._questions(2)
        html = _format_html(qs, "run-123")
        for q in qs:
            assert q["gap_id"] in html

    def test_html_is_valid_structure(self):
        qs = self._questions()
        html = _format_html(qs, "run-123")
        assert "<table" in html
        assert "</html>" in html

    def test_question_text_appears_in_plain(self):
        qs = self._questions(1)
        plain = _format_plain(qs, "run-x")
        assert qs[0]["question"] in plain


class TestSendSmeQuestionsNoConnection:

    def test_no_connection_string_marks_skipped(self):
        with patch("src.sme.email_handler.settings") as mock_s:
            mock_s.azure_communication_connection_string = ""
            qs = [{"gap_id": "x", "question": "q?", "sme_email": "s@c.com", "run_id": "r"}]
            result = send_sme_questions(qs)
        assert result[0]["send_status"] == "skipped_no_connection"

    def test_no_sme_email_marks_skipped(self):
        with patch("src.sme.email_handler.settings") as mock_s:
            mock_s.azure_communication_connection_string = "conn_str"
            mock_s.email_sender_address = "bot@corp.com"
            qs = [{"gap_id": "x", "question": "q?", "sme_email": None, "run_id": "r"}]
            result = send_sme_questions(qs)
        assert result[0]["send_status"] == "skipped_no_email"


# ===========================================================================
# Email handler — reply parsing (pure regex)
# ===========================================================================

class TestExtractRefAnswers:

    def test_single_ref_extracted(self):
        gap_id = str(uuid.uuid4())
        body = f"[REF:{gap_id}]: The data comes from ADLS bronze container"
        result = _extract_ref_answers(body)
        assert gap_id in result
        assert "ADLS bronze" in result[gap_id]

    def test_multiple_refs_extracted(self):
        id1, id2 = str(uuid.uuid4()), str(uuid.uuid4())
        body = f"[REF:{id1}]: Source is ADLS\n[REF:{id2}]: It's a SQL table"
        result = _extract_ref_answers(body)
        assert id1 in result
        assert id2 in result

    def test_case_insensitive(self):
        gap_id = str(uuid.uuid4())
        body = f"[ref:{gap_id}]: answer here"
        result = _extract_ref_answers(body)
        assert gap_id in result

    def test_no_refs_returns_empty_dict(self):
        result = _extract_ref_answers("This is a reply with no structured markers")
        assert result == {}

    def test_answer_stripped_of_whitespace(self):
        gap_id = str(uuid.uuid4())
        body = f"[REF:{gap_id}]:   trimmed answer   "
        result = _extract_ref_answers(body)
        assert result[gap_id] == "trimmed answer"


class TestExtractRunId:

    def test_run_id_extracted_from_subject(self):
        run_id = str(uuid.uuid4())
        subject = f"Re: [Lineage Agent] Run {run_id} — Input needed"
        assert _extract_run_id(subject, "") == run_id

    def test_run_id_extracted_from_body_fallback(self):
        run_id = str(uuid.uuid4())
        body = f"Run ID: [Lineage Agent] Run {run_id}"
        assert _extract_run_id("no match here", body) == run_id

    def test_returns_none_when_not_found(self):
        assert _extract_run_id("Regular reply", "No run id here") is None


class TestParseEmailResponses:

    def test_structured_reply_parsed(self):
        gap_id = str(uuid.uuid4())
        raw = [{
            "from": "sme@corp.com",
            "subject": "Re: [Lineage Agent] Run abc-123 — Input needed",
            "body": f"[REF:{gap_id}]: The source is ADLS container raw/orders",
        }]
        parsed = parse_email_responses(raw)
        assert len(parsed) == 1
        assert parsed[0]["gap_id"] == gap_id
        assert parsed[0]["sme_email"] == "sme@corp.com"
        assert "ADLS" in parsed[0]["raw_answer"]

    def test_unstructured_reply_returned_with_null_gap_id(self):
        raw = [{"from": "sme@corp.com", "subject": "Re: lineage", "body": "I don't know"}]
        parsed = parse_email_responses(raw)
        assert parsed[0]["gap_id"] is None
        assert parsed[0]["raw_answer"] == "I don't know"

    def test_multiple_refs_produce_multiple_responses(self):
        id1, id2 = str(uuid.uuid4()), str(uuid.uuid4())
        raw = [{
            "from": "sme@corp.com",
            "subject": "Re",
            "body": f"[REF:{id1}]: Source is ADLS\n[REF:{id2}]: Target is SQL",
        }]
        parsed = parse_email_responses(raw)
        assert len(parsed) == 2
        gap_ids = {p["gap_id"] for p in parsed}
        assert id1 in gap_ids and id2 in gap_ids


# ===========================================================================
# Response processor — graph updates (mock LLM)
# ===========================================================================

class TestApplyResponsesToGraph:

    def _make_structured(self, **kwargs) -> dict:
        return {
            "answer_type": "source_path",
            "primary_value": None,
            "system_type": "adls",
            "confirmed": None,
            "notes": None,
            **kwargs,
        }

    def _run(self, gaps, responses, graph, structured):
        """Helper: run apply_responses_to_graph with both LLM calls mocked."""
        with patch("src.sme.response_processor._get_llm", return_value=MagicMock()), \
             patch("src.sme.response_processor._extract_structured_answer",
                   return_value=structured):
            return apply_responses_to_graph(responses=responses, gaps=gaps, graph=graph)

    def test_missing_source_adds_new_node_and_edge(self):
        tgt = _node("silver_orders", system=SystemType.DATABRICKS, full_path="silver.orders")
        g = _graph(tgt)
        tgt_id = list(g.nodes.keys())[0]
        gap = _gap("missing_source", node_id=tgt_id)

        structured = self._make_structured(
            answer_type="source_path",
            primary_value="abfss://raw@storage.dfs.core.windows.net/orders",
            system_type="adls",
        )
        updated_gaps, updated_graph = self._run(
            [gap],
            [{"gap_id": gap.id, "sme_email": "sme@corp.com", "raw_answer": "ADLS path"}],
            g, structured,
        )
        assert any(n.system == SystemType.ADLS for n in updated_graph.nodes.values())
        assert len(updated_graph.edges) == 1
        assert updated_graph.edges[0].sme_approved_by == "sme@corp.com"
        assert updated_gaps[0].resolved is True

    def test_unknown_system_updates_node_system(self):
        n = _node("mystery", system=SystemType.UNKNOWN, full_path="mystery")
        g = _graph(n)
        node_id = list(g.nodes.keys())[0]
        gap = _gap("unknown_system", node_id=node_id)

        structured = self._make_structured(
            answer_type="system_type",
            primary_value="main.silver.mystery",
            system_type="databricks",
        )
        _, updated_graph = self._run(
            [gap],
            [{"gap_id": gap.id, "sme_email": "sme@corp.com", "raw_answer": "Databricks"}],
            g, structured,
        )
        assert updated_graph.nodes[node_id].system == SystemType.DATABRICKS

    def test_low_confidence_confirmation_sets_sme_approved(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        e = _edge(src, tgt, confidence=0.55)
        g = _graph(src, tgt, e)
        edge_id = g.edges[0].id
        gap = _gap("low_confidence_edge", edge_id=edge_id)

        structured = self._make_structured(answer_type="confirmation", confirmed=True)
        _, updated_graph = self._run(
            [gap],
            [{"gap_id": gap.id, "sme_email": "sme@corp.com", "raw_answer": "Yes confirmed"}],
            g, structured,
        )
        edge = next(e for e in updated_graph.edges if e.id == edge_id)
        assert edge.sme_approved_by == "sme@corp.com"
        assert edge.confidence == 0.95
        assert edge.confidence_level == ConfidenceLevel.CONFIRMED

    def test_low_confidence_denial_marks_edge_low(self):
        src = _node("a", full_path="a")
        tgt = _node("b", full_path="b")
        e = _edge(src, tgt, confidence=0.55)
        g = _graph(src, tgt, e)
        edge_id = g.edges[0].id
        gap = _gap("low_confidence_edge", edge_id=edge_id)

        structured = self._make_structured(
            answer_type="confirmation", confirmed=False,
            notes="Actually data comes from a different table",
        )
        _, updated_graph = self._run(
            [gap],
            [{"gap_id": gap.id, "sme_email": "sme@corp.com", "raw_answer": "No that's wrong"}],
            g, structured,
        )
        edge = next(e for e in updated_graph.edges if e.id == edge_id)
        assert edge.confidence == 0.10
        assert edge.confidence_level == ConfidenceLevel.LOW

    def test_unmatched_response_does_not_resolve_gap(self):
        # LLM is never called here because no gap matches — no need to mock _get_llm
        gap = _gap("missing_source")
        g = LineageGraph()
        unrelated_response = {"gap_id": str(uuid.uuid4()), "sme_email": "x@x.com", "raw_answer": "answer"}
        updated_gaps, _ = apply_responses_to_graph([unrelated_response], [gap], g)
        assert updated_gaps[0].resolved is False

    def test_no_information_answer_leaves_graph_unchanged(self):
        n = _node("t", full_path="t")
        g = _graph(n)
        gap = _gap("unknown_system", node_id=list(g.nodes.keys())[0])

        structured = self._make_structured(answer_type="no_information", primary_value=None)
        _, updated_graph = self._run(
            [gap],
            [{"gap_id": gap.id, "sme_email": "s@c.com", "raw_answer": "I don't know"}],
            g, structured,
        )
        assert updated_graph.nodes[list(g.nodes.keys())[0]].system == SystemType.DATABRICKS

    def test_duplicate_edge_not_created_when_sme_adds_existing(self):
        src = _node("a", system=SystemType.ADLS, full_path="adls://a")
        tgt = _node("b", system=SystemType.DATABRICKS, full_path="db.b")
        e = _edge(src, tgt, confidence=0.55)
        g = _graph(src, tgt, e)
        tgt_id = next(nid for nid, n in g.nodes.items() if n.full_path == "db.b")
        gap = _gap("missing_source", node_id=tgt_id)

        structured = self._make_structured(
            answer_type="source_path", primary_value="adls://a", system_type="adls",
        )
        _, updated_graph = self._run(
            [gap],
            [{"gap_id": gap.id, "sme_email": "s@c.com", "raw_answer": "ADLS"}],
            g, structured,
        )
        assert len(updated_graph.edges) == 1


# ===========================================================================
# Helpers
# ===========================================================================

class TestHelpers:

    def test_parse_system_valid(self):
        assert _parse_system("databricks") == SystemType.DATABRICKS
        assert _parse_system("adls") == SystemType.ADLS
        assert _parse_system("sql") == SystemType.SQL

    def test_parse_system_invalid_returns_unknown(self):
        assert _parse_system("gibberish") == SystemType.UNKNOWN
        assert _parse_system(None) == SystemType.UNKNOWN

    def test_find_or_create_node_creates_new(self):
        g = LineageGraph()
        node = _find_or_create_node("abfss://raw/orders", SystemType.ADLS, g)
        assert node.system == SystemType.ADLS
        assert node.id in g.nodes

    def test_find_or_create_node_returns_existing(self):
        g = LineageGraph()
        n1 = _find_or_create_node("some.table", SystemType.DATABRICKS, g)
        n2 = _find_or_create_node("some.table", SystemType.DATABRICKS, g)
        assert n1.id == n2.id
        assert len(g.nodes) == 1
