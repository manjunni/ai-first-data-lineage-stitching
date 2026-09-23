"""
Tests for UI layer: api_client (with mocked requests) and pure component helpers.
No Streamlit rendering is tested — only the logic layer.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from src.ui.api_client import LineageApiClient, ApiError
from src.ui.components import (
    status_badge, confidence_badge, gap_type_badge,
    _STATUS_COLOUR, _CONFIDENCE_COLOUR, _GAP_TYPE_ICON,
    PAGE_DESCRIPTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client(base_url="http://localhost:8000") -> LineageApiClient:
    return LineageApiClient(base_url)


def _mock_response(status_code: int, body) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
    if status_code >= 400:
        from requests import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===========================================================================
# ApiClient — URL construction
# ===========================================================================

class TestApiClientUrls:

    def test_trailing_slash_stripped(self):
        c = LineageApiClient("http://api.example.com/")
        assert c.base_url == "http://api.example.com"

    def test_health_calls_correct_path(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {"status": "ok"})) as mock_get:
            c.health()
        mock_get.assert_called_once()
        url = mock_get.call_args[0][0]
        assert url.endswith("/health")

    def test_list_runs_no_filter(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, [])) as mock_get:
            c.list_runs()
        url = mock_get.call_args[0][0]
        assert url.endswith("/runs")

    def test_list_runs_with_status_filter(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, [])) as mock_get:
            c.list_runs(status="emails_sent")
        _, kwargs = mock_get.call_args
        assert kwargs.get("params", {}).get("status") == "emails_sent"

    def test_get_run_uses_run_id(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {"run_id": "abc"})) as mock_get:
            c.get_run("abc-123")
        url = mock_get.call_args[0][0]
        assert "abc-123" in url

    def test_get_lineage_path(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {})) as mock_get:
            c.get_lineage("run-99")
        assert "lineage" in mock_get.call_args[0][0]

    def test_get_gaps_resolved_filter(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {})) as mock_get:
            c.get_gaps("run-1", resolved=False)
        params = mock_get.call_args[1].get("params", {})
        assert params.get("resolved") == "false"

    def test_get_questions_answered_filter(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {})) as mock_get:
            c.get_questions("run-1", answered=True)
        params = mock_get.call_args[1].get("params", {})
        assert params.get("answered") == "true"

    def test_publish_preview_sends_catalog_param(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {})) as mock_get:
            c.publish_preview("run-1", catalog="collibra")
        params = mock_get.call_args[1].get("params", {})
        assert params.get("catalog") == "collibra"


# ===========================================================================
# ApiClient — response handling
# ===========================================================================

class TestApiClientResponses:

    def test_get_returns_json_body(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(200, {"status": "ok"})):
            result = c.health()
        assert result == {"status": "ok"}

    def test_404_raises_api_error_with_code(self):
        c = _client()
        with patch.object(c._session, "get", return_value=_mock_response(404, {"detail": "not found"})):
            with pytest.raises(ApiError) as exc_info:
                c.get_run("missing")
        assert exc_info.value.status_code == 404

    def test_post_sends_json_body(self):
        c = _client()
        with patch.object(c._session, "post", return_value=_mock_response(200, {"run_id": "x"})) as mock_post:
            c.publish("run-1", catalog="alation")
        _, kwargs = mock_post.call_args
        assert kwargs.get("json", {}).get("catalog") == "alation"

    def test_submit_responses_posts_to_correct_path(self):
        c = _client()
        with patch.object(c._session, "post", return_value=_mock_response(200, {})) as mock_post:
            c.submit_responses("run-xyz", [{"gap_id": "g1", "raw_answer": "a"}])
        url = mock_post.call_args[0][0]
        assert "run-xyz" in url
        assert "responses" in url

    def test_submit_response_files_posts_multipart_payload(self):
        c = _client()
        import io
        files = [("files", ("response.txt", io.BytesIO(b"[REF:g1]: answer"), "text/plain"))]
        with patch.object(c._session, "post", return_value=_mock_response(200, {"status": "ok"})) as mock_post:
            result = c.submit_response_files(
                "run-xyz", files, text="[REF:g2]: extra", sme_email="sme@example.com"
            )
        assert result == {"status": "ok"}
        url = mock_post.call_args[0][0]
        kwargs = mock_post.call_args[1]
        assert url.endswith("/runs/run-xyz/responses/files")
        assert kwargs["files"] == files
        assert kwargs["data"] == {
            "text": "[REF:g2]: extra",
            "sme_email": "sme@example.com",
        }

    def test_submit_response_files_omits_blank_optional_fields(self):
        c = _client()
        with patch.object(c._session, "post", return_value=_mock_response(200, {})) as mock_post:
            c.submit_response_files("run-xyz", [], text=" ", sme_email="")
        assert mock_post.call_args[1]["data"] == {}

    def test_connection_error_raises_api_error_status_0(self):
        c = _client()
        import requests
        with patch.object(c._session, "get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(ApiError) as exc_info:
                c.health()
        assert exc_info.value.status_code == 0

    def test_start_run_posts_multipart(self):
        c = _client()
        import io
        files = [("files", ("test.sql", io.BytesIO(b"SELECT 1"), "text/plain"))]
        with patch.object(c._session, "post", return_value=_mock_response(201, {"run_id": "new"})) as mock_post:
            result = c.start_run(files, catalog="dry_run")
        assert result == {"run_id": "new"}
        # multipart uses files= kwarg
        _, kwargs = mock_post.call_args
        assert "files" in kwargs


# ===========================================================================
# Components — pure helper functions
# ===========================================================================

class TestStatusBadge:

    def test_known_status_returns_html(self):
        badge = status_badge("published")
        assert "published" in badge
        assert "<span" in badge

    def test_unknown_status_does_not_crash(self):
        badge = status_badge("alien_status")
        assert "alien_status" in badge

    def test_all_known_statuses_have_entries(self):
        for status in _STATUS_COLOUR:
            badge = status_badge(status)
            assert status in badge


class TestConfidenceBadge:

    def test_returns_html_with_percentage(self):
        badge = confidence_badge("high", 0.90)
        assert "90%" in badge
        assert "<span" in badge

    def test_confirmed_uses_correct_colour(self):
        badge = confidence_badge("confirmed", 0.95)
        colour = _CONFIDENCE_COLOUR["confirmed"]
        assert colour in badge

    def test_low_uses_red_colour(self):
        badge = confidence_badge("low", 0.40)
        assert _CONFIDENCE_COLOUR["low"] in badge

    def test_unknown_level_does_not_crash(self):
        badge = confidence_badge("experimental", 0.75)
        assert "75%" in badge


class TestGapTypeBadge:

    def test_known_type_includes_icon(self):
        badge = gap_type_badge("missing_source")
        assert _GAP_TYPE_ICON["missing_source"] in badge

    def test_underscore_replaced_with_space(self):
        badge = gap_type_badge("missing_source")
        assert "missing source" in badge

    def test_unknown_type_does_not_crash(self):
        badge = gap_type_badge("new_gap_type")
        assert "new gap type" in badge

    def test_all_known_gap_types_have_icons(self):
        for gap_type in _GAP_TYPE_ICON:
            badge = gap_type_badge(gap_type)
            assert _GAP_TYPE_ICON[gap_type] in badge


class TestPageDescriptions:

    NAVIGATION_PAGES = {
        "Dashboard",
        "New Run",
        "Lineage Graph",
        "Gaps & Questions",
        "Submit Responses",
        "Publish",
        "Audit Trail",
    }

    def test_every_navigation_page_has_a_description(self):
        assert set(PAGE_DESCRIPTIONS) == self.NAVIGATION_PAGES
        assert all(description.strip() for description in PAGE_DESCRIPTIONS.values())

    def test_descriptions_are_approximately_twenty_words(self):
        for description in PAGE_DESCRIPTIONS.values():
            word_count = len(description.split())
            assert 15 <= word_count <= 25
