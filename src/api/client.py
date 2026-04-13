"""
src/api/client.py

Async OpenRouter client.
All API calls in the framework go through this module.
Handles authentication, retries, and error normalisation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp

from src.api.rate_limiter import RateLimiter
from src.config import ApiConfig, GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ApiError(Exception):
    """Raised when the API returns a non-recoverable error."""
    def __init__(self, status: int, message: str, model: str) -> None:
        self.status = status
        self.model = model
        super().__init__(f"API error {status} for model '{model}': {message}")


class RateLimitError(ApiError):
    """Raised on 429 — caller may retry after backoff."""
    pass


class ApiTimeoutError(Exception):
    """Raised when a request exceeds timeout_seconds."""
    pass


# ---------------------------------------------------------------------------
# Message helper
# ---------------------------------------------------------------------------

def build_messages(
    system: Optional[str],
    history: list[dict[str, str]],
    user_message: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Construct the messages list expected by the OpenRouter chat API.

    Args:
        system:       System prompt string, or None to omit.
        history:      Existing list of {"role": ..., "content": ...} dicts.
        user_message: Optional new user message to append.

    Returns:
        Full messages list ready to send.
    """
    messages: list[dict[str, str]] = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.extend(history)

    if user_message is not None:
        messages.append({"role": "user", "content": user_message})

    return messages


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """
    Thin async wrapper around the OpenRouter /v1/chat/completions endpoint.

    One instance is created in main.py and shared across all agents.
    The underlying aiohttp.ClientSession is opened once and reused.

    Usage:
        async with OpenRouterClient(api_config, generation_config) as client:
            response = await client.complete(
                model="x-ai/grok-3",
                messages=[...],
                temperature=0.7,
                max_tokens=300,
            )
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_config: ApiConfig,
        generation_config: GenerationConfig,
        rate_limiter: RateLimiter,
    ) -> None:
        self._api_config = api_config
        self._gen_config = generation_config
        self._rate_limiter = rate_limiter
        self._session: Optional[aiohttp.ClientSession] = None
        self._api_key = os.environ["OPENROUTER_API_KEY"]  # validated in config.py

    async def __aenter__(self) -> OpenRouterClient:
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/adversarial-debate",  # OpenRouter best practice
            },
            timeout=aiohttp.ClientTimeout(total=self._api_config.timeout_seconds),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Send a chat completion request, return the response content string.

        Retries on rate limit (429) and transient server errors (5xx)
        with exponential backoff. Raises ApiError on unrecoverable failure.
        """
        if self._session is None:
            raise RuntimeError("Client must be used as an async context manager")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self._api_config.max_retries + 1):
            try:
                async with self._rate_limiter.acquire(model):
                    content = await self._post(model, payload)
                    return content

            except RateLimitError as e:
                last_error = e
                wait = self._api_config.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Rate limit hit for model '%s' (attempt %d/%d). "
                    "Retrying in %.1fs.",
                    model, attempt + 1, self._api_config.max_retries, wait,
                )
                await asyncio.sleep(wait)

            except ApiTimeoutError as e:
                last_error = e
                wait = self._api_config.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Timeout for model '%s' (attempt %d/%d). "
                    "Retrying in %.1fs.",
                    model, attempt + 1, self._api_config.max_retries, wait,
                )
                await asyncio.sleep(wait)

            except ApiError:
                # Non-recoverable (4xx other than 429) — do not retry
                raise

        raise last_error  # type: ignore[misc]

    async def _post(self, model: str, payload: dict[str, Any]) -> str:
        """Single POST attempt. Raises appropriate exceptions on failure."""
        assert self._session is not None

        try:
            async with self._session.post(self.BASE_URL, json=payload) as resp:
                if resp.status == 429:
                    body = await resp.json()
                    raise RateLimitError(429, body.get("error", {}).get("message", ""), model)

                if resp.status >= 500:
                    body = await resp.text()
                    raise RateLimitError(resp.status, body, model)
                    # Treat 5xx as retryable by reusing RateLimitError

                if resp.status >= 400:
                    body = await resp.json()
                    raise ApiError(
                        resp.status,
                        body.get("error", {}).get("message", str(body)),
                        model,
                    )

                data = await resp.json()
                return _extract_content(data, model)

        except asyncio.TimeoutError as e:
            raise ApiTimeoutError(f"Request timed out for model '{model}'") from e

        except aiohttp.ClientError as e:
            raise ApiError(0, f"Network error: {e}", model) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_content(data: dict[str, Any], model: str) -> str:
    """
    Pull the assistant message content out of an OpenRouter response.
    Raises ApiError if the response structure is unexpected.
    """
    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ApiError(0, f"Unexpected content type: {type(content)}", model)
        return content.strip()
    except (KeyError, IndexError) as e:
        raise ApiError(0, f"Unexpected response structure: {data}", model) from e