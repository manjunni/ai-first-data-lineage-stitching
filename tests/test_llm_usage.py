import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agents.lineage_agent import parse_artifacts, stitch_graph_node
from src.graph.models import Gap, LineageGraph
from src.llm_client import create_chat_model
from src.llm_usage import extract_token_usage
from src.parsers.sql_parser import SqlParser
from src.sme.question_generator import generate_questions
from src.sme.response_processor import apply_responses_to_graph


def test_chat_model_uses_openai_compatible_endpoint_settings():
    with patch("src.llm_client.ChatOpenAI") as chat_openai, \
         patch("src.llm_client.settings.azure_openai_endpoint", "https://example.test/v1"), \
         patch("src.llm_client.settings.azure_openai_api_key", "test-key"), \
         patch("src.llm_client.settings.azure_openai_deployment", "test-model"):
        create_chat_model(temperature=0.2)

    chat_openai.assert_called_once_with(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        temperature=0.2,
    )


def test_extract_token_usage_supports_langchain_and_openai_metadata():
    assert extract_token_usage(SimpleNamespace(usage_metadata={
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    })) == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}

    assert extract_token_usage(SimpleNamespace(response_metadata={
        "token_usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    })) == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


def test_extract_token_usage_marks_missing_counts_unavailable():
    assert extract_token_usage(SimpleNamespace(content="answer")) == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_parse_artifacts_records_deterministic_parser_and_stitch_modes():
    parsed = parse_artifacts({"artifacts": [{
        "filename": "orders.sql",
        "content": "CREATE TABLE target AS SELECT * FROM source",
    }]})

    parse_usage = next(
        event for event in parsed["audit_trail"]
        if event.get("step") == "usage" and event.get("stage") == "parse"
    )
    assert parse_usage["execution_mode"] == "deterministic"
    assert parse_usage["llm_called"] is False

    stitched = stitch_graph_node({"fragments": parsed["fragments"]})
    stitch_usage = next(
        event for event in stitched["audit_trail"] if event.get("stage") == "stitch"
    )
    assert stitch_usage["execution_mode"] == "deterministic"
    assert stitch_usage["llm_called"] is False


def test_sql_fallback_records_llm_token_usage():
    response = SimpleNamespace(
        content='{"relationships": []}',
        usage_metadata={"input_tokens": 23, "output_tokens": 5, "total_tokens": 28},
    )
    llm = MagicMock()
    llm.invoke.return_value = response
    with patch("src.agents.lineage_agent._PARSERS", [SqlParser()]), \
         patch("src.parsers.sql_parser._parse_with_sqlglot", side_effect=ValueError), \
         patch("src.parsers.sql_parser.create_chat_model", return_value=llm):
        result = parse_artifacts({"artifacts": [{
            "filename": "broken.sql",
            "content": "invalid",
        }]})

    usage = next(
        event for event in result["audit_trail"]
        if event.get("step") == "usage" and event.get("stage") == "parse"
    )
    assert usage["execution_mode"] == "llm"
    assert usage["outcome"] == "success"
    assert usage["llm_called"] is True
    assert usage["input_tokens"] == 23
    assert usage["output_tokens"] == 5


def test_question_generation_reports_llm_usage():
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
    )
    gap = Gap(gap_type="missing_source", description="Missing upstream source")
    events = []
    with patch("src.sme.question_generator._get_llm", return_value=MagicMock()), \
         patch("src.sme.question_generator._llm_enrich", return_value=("Enriched", response)):
        questions = generate_questions([gap], LineageGraph(), usage_callback=events.append)

    assert questions[0]["question"] == "Enriched"
    assert events[0]["execution_mode"] == "llm"
    assert events[0]["input_tokens"] == 9
    assert events[0]["output_tokens"] == 6


def test_question_generation_reports_deterministic_usage():
    gap = Gap(gap_type="missing_source", description="Missing upstream source")
    events = []
    generate_questions(
        [gap], LineageGraph(), enrich_with_llm=False, usage_callback=events.append,
    )

    assert events[0]["execution_mode"] == "deterministic"
    assert events[0]["llm_called"] is False


def test_sme_response_reports_llm_usage():
    response = SimpleNamespace(
        content=json.dumps({
            "answer_type": "no_information",
            "primary_value": None,
            "system_type": "unknown",
            "confirmed": None,
            "notes": None,
        }),
        usage_metadata={"input_tokens": 17, "output_tokens": 4, "total_tokens": 21},
    )
    llm = MagicMock()
    llm.invoke.return_value = response
    gap = Gap(gap_type="missing_source", description="Missing upstream source")
    events = []
    with patch("src.sme.response_processor._get_llm", return_value=llm):
        apply_responses_to_graph(
            [{"gap_id": gap.id, "raw_answer": "I do not know"}],
            [gap],
            LineageGraph(),
            usage_callback=events.append,
        )

    assert events[0]["execution_mode"] == "llm"
    assert events[0]["input_tokens"] == 17
    assert events[0]["output_tokens"] == 4


def test_sme_response_without_llm_records_deterministic_processing():
    gap = Gap(gap_type="stewardship_required", description="Identify the external producer")
    events = []
    with patch("src.sme.response_processor._get_llm", side_effect=RuntimeError):
        apply_responses_to_graph(
            [{"gap_id": gap.id, "raw_answer": "I don't know"}],
            [gap],
            LineageGraph(),
            usage_callback=events.append,
        )

    assert events[0]["execution_mode"] == "deterministic"
    assert events[0]["llm_called"] is False
