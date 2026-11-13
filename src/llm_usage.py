from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import settings


def extract_token_usage(response: Any) -> dict[str, Optional[int]]:
    """Normalize token counts reported by LangChain or the OpenAI response."""
    metadata = getattr(response, "usage_metadata", None)
    if not isinstance(metadata, Mapping):
        response_metadata = getattr(response, "response_metadata", None)
        metadata = (
            response_metadata.get("token_usage")
            if isinstance(response_metadata, Mapping)
            else None
        )

    if not isinstance(metadata, Mapping):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    def count(*keys: str) -> Optional[int]:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    return {
        "input_tokens": count("input_tokens", "prompt_tokens"),
        "output_tokens": count("output_tokens", "completion_tokens"),
        "total_tokens": count("total_tokens"),
    }


def make_usage_event(
    stage: str,
    execution_mode: str,
    outcome: str,
    *,
    response: Any = None,
    llm_called: bool = False,
    artifact: Optional[str] = None,
    gap_id: Optional[str] = None,
    error_type: Optional[str] = None,
) -> dict:
    usage = extract_token_usage(response) if response is not None else {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    return {
        "step": "usage",
        "stage": stage,
        "execution_mode": execution_mode,
        "outcome": outcome,
        "llm_called": llm_called,
        "model": settings.azure_openai_deployment if llm_called else None,
        **usage,
        "artifact": artifact,
        "gap_id": gap_id,
        "error_type": error_type,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
