"""
src/api/client.py

Async OpenRouter client.
All API calls in the framework go through this module.
Handles authentication, retries, and error normalisation.

End-to-end flow:
1. Build chat messages from system prompt, history, and optional new user input.
2. Open one shared HTTP session with API key and timeout settings.
3. Send the request through a per-model rate limiter.
4. Retry on temporary failures (rate limits, timeouts, server errors) with backoff.
5. Parse the response and return:
   - content: final assistant text
   - reasoning: optional model reasoning text
6. Convert API/network/format errors into consistent custom exceptions.
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
    def __init__(self, status: int, message: str, model: str, retry_after: float | None = None) -> None:
        super().__init__(status, message, model)
        self.retry_after = retry_after


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

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

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
        self._base_url = api_config.base_url or self.DEFAULT_BASE_URL

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
    ) -> tuple[str, str | None]:
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
                    content, reasoning = await self._post(model, payload)
                    return content, reasoning

            except RateLimitError as e:
                last_error = e
                wait = e.retry_after if e.retry_after is not None \
                    else self._api_config.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Rate limit hit for model '%s' (attempt %d/%d). "
                    "Retrying in %.1fs.%s",
                    model, attempt + 1, self._api_config.max_retries + 1, wait,
                    " (Retry-After header)" if e.retry_after else " (exponential backoff)",
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

        if last_error is not None:
            raise last_error

        raise RuntimeError("Request failed without a captured exception")  # type: ignore[misc]
    
    async def _post(self, model: str, payload: dict[str, Any]) -> tuple[str, str | None]:
        """
        Send one chat completion request with retry handling.

        Returns:
            tuple[str, str | None]:
                - content: model response text
                - reasoning: optional reasoning text if provided by the model

        Retries:
            - RateLimitError (429) and transient server-side failures
            - ApiTimeoutError
        Raises:
            - ApiError for non-recoverable API failures
        """
        assert self._session is not None

        try:
            async with self._session.post(self._base_url, json=payload) as resp:
                if resp.status == 429:
                    retry_after = _parse_retry_after(resp.headers)
                    message = await _read_error_message(resp)
                    raise RateLimitError(429, message, model, retry_after=retry_after)

                if resp.status >= 500:
                    message = await _read_error_message(resp)
                    raise RateLimitError(resp.status, message, model)

                if resp.status >= 400:
                    message = await _read_error_message(resp)
                    raise ApiError(resp.status, message, model)

                data = await resp.json()

                choice0 = data.get("choices", [{}])[0]
                message = choice0.get("message", {})
                # logger.warning(
                #     "DEBUG OpenRouter response for %s: finish_reason=%r, message_keys=%r, content_type=%r, message=%r",
                #     model,
                #     choice0.get("finish_reason"),
                #     list(message.keys()) if isinstance(message, dict) else None,
                #     type(message.get("content")) if isinstance(message, dict) else None,
                #     message,
                # )
                return _extract_content(data, model)

        except asyncio.TimeoutError as e:
            raise ApiTimeoutError(f"Request timed out for model '{model}'") from e

        except aiohttp.ClientError as e:
            raise ApiError(0, f"Network error: {e}", model) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _read_error_message(resp: aiohttp.ClientResponse) -> str:
    try:
        data = await resp.json(content_type=None)
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message", data))
            return str(data)
        return str(data)
    except Exception:
        try:
            return await resp.text()
        except Exception:
            return f"HTTP {resp.status}"
        

def _extract_content(data: dict[str, Any], model: str) -> tuple[str, str | None]:
    # Handle OpenRouter gateway errors returned as 200 with error body
    if "error" in data and "choices" not in data:
        error_msg = data["error"].get("message", str(data["error"])) if isinstance(data["error"], dict) else str(data["error"])
        error_code = data["error"].get("code", 0) if isinstance(data["error"], dict) else 0
        if isinstance(error_code, int) and error_code >= 500:
            raise RateLimitError(error_code, error_msg, model)
        raise ApiError(error_code, error_msg, model)

    try:
        message = data["choices"][0]["message"]
        content = message.get("content")
        reasoning = message.get("reasoning") or None  # None if absent or empty string

        if content is None:
            refusal = message.get("refusal")
            if refusal:
                raise ApiError(0, f"Model refused: {refusal}", model)
            raise ApiError(0, "Content is None with no refusal message", model)

        if not isinstance(content, str):
            raise ApiError(0, f"Unexpected content type: {type(content)}", model)

        return content.strip(), reasoning
    except (KeyError, IndexError) as e:
        raise ApiError(0, f"Unexpected response structure: {data}", model) from e
    
def _parse_retry_after(headers) -> float | None:
    """
    Parse Retry-After header from a 429 response.
    Returns seconds to wait as float, or None if header absent or unparseable.
    OpenRouter may return seconds (int) or an HTTP date — we handle both.
    """
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        # Could be an HTTP date string — fall back to fixed backoff
        logger.debug("Unparseable Retry-After header: %r", value)
        return None