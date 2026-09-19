"""
Generator: query → retrieve → prompt LLM → answer with citations.

Wraps the Retriever and a generation LLM. Reuses the retry and error
infrastructure from src/llm/ so provider hiccups are handled the same
way everywhere in the pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv

from catalog.llm_models import LLMConfig
from generation.prompts import SYSTEM_PROMPT, build_user_prompt
from llm import LLMError, RetryExhausted, retry_call
from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

load_dotenv()


# ─── Data shapes ───────────────────────────────────────────────────── #


@dataclass
class Citation:
    index: int
    book: str
    page: int
    section: str = ""


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)


# ─── Gemini adapter ────────────────────────────────────────────────── #


def _make_gemini_generator(model_name: str) -> Callable[[str, str], str]:
    """Return a callable(system, user) -> str that queries Gemini."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env or export it in your shell."
        )

    client = genai.Client(api_key=api_key)

    def generate(system: str, user: str) -> str:
        def _do_call() -> str:
            response = client.models.generate_content(
                model=model_name,
                contents=user,
                config={
                    "temperature": 0.2,
                    "system_instruction": system,
                },
            )
            return (response.text or "").strip()

        return retry_call(_do_call, provider="gemini", max_attempts=5)

    return generate


# ─── Generator ─────────────────────────────────────────────────────── #


class Generator:
    """End-to-end RAG generator: query → chunks → answer."""

    def __init__(
        self,
        retriever: Retriever,
        model_key: str = "gemini-flash",
        top_k: int = 5,
    ) -> None:
        """
        Args:
            retriever: A configured Retriever instance.
            model_key: Catalog key for the generation model.
            top_k: How many chunks to retrieve per query.
        """
        self.retriever = retriever
        self.top_k = top_k

        cfg = LLMConfig.get_config(model_key)
        self.model_key = model_key
        self.model_name = cfg["name"]
        provider = cfg["provider"]

        logger.info(f"Generator model: {model_key} → {provider}/{self.model_name}")

        if provider != "gemini":
            raise ValueError(f"Unsupported generator provider: {provider}")

        self._generate = _make_gemini_generator(self.model_name)

    # ─── Public API ─────────────────────────────────────────────────── #

    def answer(self, query: str) -> GeneratedAnswer:
        """
        Retrieve chunks for the query and generate an answer with citations.
        """
        if not query.strip():
            return GeneratedAnswer(query=query, answer="", citations=[], chunks=[])

        chunks = self.retriever.search(query, k=self.top_k)
        if not chunks:
            return GeneratedAnswer(
                query=query,
                answer="I don't have enough information in the provided sources to answer this.",
                citations=[],
                chunks=[],
            )

        user_prompt = build_user_prompt(query, chunks)

        try:
            raw = self._generate(SYSTEM_PROMPT, user_prompt)
        except (LLMError, RetryExhausted) as e:
            logger.error(f"Generation failed: {e}")
            return GeneratedAnswer(
                query=query,
                answer=f"[generation failed: {e}]",
                citations=[],
                chunks=chunks,
            )

        citations = [
            Citation(
                index=i,
                book=chunk.get("book", ""),
                page=int(chunk.get("page", 0) or 0),
                section=chunk.get("section", "") or "",
            )
            for i, chunk in enumerate(chunks, start=1)
        ]

        return GeneratedAnswer(
            query=query,
            answer=raw,
            citations=citations,
            chunks=chunks,
        )