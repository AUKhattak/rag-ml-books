"""
Retry helper with exponential backoff and rate-limit awareness.

Used by provider adapters to wrap API calls. Provider-agnostic — it
just calls classify_provider_error to decide whether to retry.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

from llm.errors import (
    LLMError,
    classify_provider_error,
    is_daily_quota_error,
    is_rate_limit_status,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""


def retry_call(
    fn: Callable[[], T],
    *,
    provider: str = "unknown",
    max_attempts: int = 5,
    base_delay: float = 2.0,
    rate_limit_delay: float = 20.0,
    max_delay: float = 120.0,
) -> T:
    """
    Call fn() and retry on transient errors.

    Args:
        fn:                Zero-arg callable that performs the API call.
        provider:          For error classification.
        max_attempts:      Total number of attempts, including the first.
        base_delay:        Base delay for generic retryable errors (500s).
        rate_limit_delay:  Base delay for 429s (usually need to wait longer).
        max_delay:         Ceiling on any single sleep.

    Returns:
        Whatever fn() returns on success.

    Raises:
        RetryExhausted:   If max_attempts are used and all fail.
        LLMError:         If a non-retryable error occurs, or if the daily
                          quota is exhausted.
    """
    last_err: LLMError | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            err = classify_provider_error(exc, provider=provider)

            # Daily quota exhaustion: no point retrying.
            if is_daily_quota_error(err):
                logger.error(
                    f"[{provider}] daily quota exhausted. "
                    f"Quota resets at the provider's midnight. "
                    f"Consider switching to another model or waiting."
                )
                raise err from exc

            # Other non-retryable errors.
            if not err.retryable:
                logger.error(f"Non-retryable error: {err}")
                raise err from exc

            if attempt == max_attempts:
                logger.error(f"Retry exhausted after {max_attempts} attempts: {err}")
                last_err = err
                break

            if is_rate_limit_status(err.status_code):
                delay = min(rate_limit_delay * attempt, max_delay)
            else:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            logger.warning(
                f"[{provider}] attempt {attempt}/{max_attempts} failed "
                f"({err.status_code}); sleeping {delay:.0f}s"
            )
            time.sleep(delay)

    raise RetryExhausted(str(last_err)) from last_err