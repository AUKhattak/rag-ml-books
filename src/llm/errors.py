"""
Error classification for LLM provider calls.

Providers (Gemini, OpenAI, Anthropic) return HTTP-style errors that we
need to distinguish into:
    - retryable (transient): rate limits, server errors, timeouts
    - non-retryable (permanent): bad requests, auth failures, missing models

This module normalizes those into a small set of predicates so calling
code doesn't have to know provider-specific error shapes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMError(Exception):
    """Generic LLM error with classification info."""

    message: str
    status_code: int | None = None
    retryable: bool = False
    provider: str = "unknown"

    def __str__(self) -> str:
        return f"[{self.provider} status={self.status_code}] {self.message}"


# ─── Retryable status codes ────────────────────────────────────────── #

RETRYABLE_STATUS = {
    408,  # Request Timeout
    409,  # Conflict (rare, but sometimes transient)
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}


def is_retryable_status(status: int | None) -> bool:
    return status in RETRYABLE_STATUS


def is_rate_limit_status(status: int | None) -> bool:
    return status == 429


def is_server_error_status(status: int | None) -> bool:
    return status is not None and 500 <= status < 600


# ─── Provider-agnostic classifier ──────────────────────────────────── #


def classify_provider_error(exc: Exception, provider: str = "unknown") -> LLMError:
    """
    Turn a provider-specific exception into an LLMError with a
    retryable flag.

    Works with google.genai errors. Extend this function when adding
    other providers (OpenAI, Anthropic).
    """
    status: int | None = None
    message = str(exc)

    # ─── Google Gemini ─────────────────────────────────────────────── #
    if provider == "gemini":
        # google.genai.errors.ClientError / ServerError carry status codes.
        try:
            from google.genai import errors as gerrors

            if isinstance(exc, gerrors.ClientError | gerrors.ServerError):
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        except Exception:  # noqa: BLE001
            pass

        # Fallback: parse status from the string if the attribute is absent.
        if status is None:
            for candidate in (429, 500, 502, 503, 504, 404, 401, 403, 400):
                if str(candidate) in message:
                    status = candidate
                    break

        retryable = is_retryable_status(status)

        return LLMError(
            message=message,
            status_code=status,
            retryable=retryable,
            provider=provider,
        )

    # Unknown provider: conservatively not retryable.
    return LLMError(message=message, provider=provider, retryable=False)