from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from config.settings import settings


def create_chat_model(temperature: Optional[float] = None) -> ChatOpenAI:
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    return ChatOpenAI(
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        model=settings.azure_openai_deployment,
        **options,
    )
