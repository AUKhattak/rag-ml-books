"""
LLM-as-judge for retrieval evaluation.

Resolves the model via the catalog, calls the provider with retry-on-
transient-error, and caches responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from catalog.llm_models import LLMConfig
from llm import LLMError, RetryExhausted, retry_call

logger = logging.getLogger(__name__)

load_dotenv()

# ─── Cache ─────────────────────────────────────────────────────────── #

_CACHE_DIR = Path("data/evaluation/judge_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]


def _cache_get(prompt: str) -> str | None:
    path = _CACHE_DIR / f"{_cache_key(prompt)}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)["response"]
    return None


def _cache_set(prompt: str, response: str) -> None:
    path = _CACHE_DIR / f"{_cache_key(prompt)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt[:500], "response": response}, f, ensure_ascii=False)


# ─── Gemini adapter ────────────────────────────────────────────────── #


def _make_gemini_judge(model_name: str) -> Callable[[str], str]:
    """Create a judge callable that queries Gemini with retry + throttle."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env or export it in your shell."
        )

    client = genai.Client(api_key=api_key)

    # Free tier: 15 req/min. ~4.5s between calls keeps us under.
    min_interval = 4.5
    state = {"last_call": 0.0}

    def judge(prompt: str) -> str:
        cached = _cache_get(prompt)
        if cached is not None:
            return cached

        def _do_call() -> str:
            # Pre-call throttle.
            now = time.time()
            wait = min_interval - (now - state["last_call"])
            if wait > 0:
                time.sleep(wait)

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.0},
            )
            state["last_call"] = time.time()
            return (response.text or "").strip()

        try:
            text = retry_call(_do_call, provider="gemini", max_attempts=5)
        except (LLMError, RetryExhausted) as e:
            logger.error(f"Judge call failed: {e}")
            raise

        _cache_set(prompt, text)
        return text

    return judge


# ─── Public factory ────────────────────────────────────────────────── #


def make_judge(model_key: str = "gemini-flash-lite") -> Callable[[str], str]:
    """Return a judge callable for the given catalog model key."""
    cfg = LLMConfig.get_config(model_key)
    provider = cfg["provider"]
    model_name = cfg["name"]

    logger.info(f"Judge: {model_key} → {provider}/{model_name}")

    if provider == "gemini":
        return _make_gemini_judge(model_name)

    raise ValueError(f"Unknown provider: {provider}")


# ─── The judge decision ────────────────────────────────────────────── #


JUDGE_TEMPLATE = """You are evaluating retrieval quality for a RAG system on machine learning textbooks.

Query:
{query}

Expected answer (summary):
{expected}

Retrieved chunk:
{chunk}

Does the retrieved chunk contain information that answers the query?
Answer with a single word: YES or NO. No explanation."""


def chunk_answers_query(
    chunk_content: str,
    query: str,
    expected_answer_summary: str,
    judge: Callable[[str], str],
) -> bool:
    """Ask the judge whether the chunk answers the query. Returns bool."""
    prompt = JUDGE_TEMPLATE.format(
        query=query,
        expected=expected_answer_summary,
        chunk=chunk_content[:2000],
    )
    response = judge(prompt).strip().upper()
    return response.startswith("YES")