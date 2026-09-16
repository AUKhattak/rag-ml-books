"""Catalog of LLM models used for judging, generation, and query rewriting."""

from __future__ import annotations

from typing import Any


class LLMConfig:
    """Configuration for LLM models."""

    MODELS: dict[str, dict[str, Any]] = {
        "gemini-flash-lite": {
            "provider": "gemini",
            "name": "gemini-3.5-flash-lite",
            "description": "Cheapest Gemini; good for YES/NO judging.",
        },
        "gemini-flash": {
            "provider": "gemini",
            "name": "gemini-3.6-flash",
            "description": "Fast Gemini; good default for generation.",
        },
        "gemini-pro": {
            "provider": "gemini",
            "name": "gemini-3.6-pro",
            "description": "Higher quality; use for final answer generation.",
        },
    }

    @classmethod
    def get_config(cls, model_key: str) -> dict[str, Any]:
        if model_key not in cls.MODELS:
            raise ValueError(
                f"Unknown LLM model key: {model_key}. "
                f"Available: {list(cls.MODELS.keys())}"
            )
        return cls.MODELS[model_key]

    @classmethod
    def get_available_models(cls) -> list[str]:
        return list(cls.MODELS.keys())