# """
# src/api/rate_limiter.py

# Async token bucket rate limiter.
# One bucket per model, sized by concurrency_limits in experiment.yaml.

# Usage:
#     limiter = RateLimiter(config.api.concurrency_limits)
#     async with limiter.acquire(model_name):
#         response = await call_api(...)
# """

# from __future__ import annotations

# import asyncio
# from contextlib import asynccontextmanager


# class RateLimiter:
#     """
#     Manages a semaphore per model to cap concurrent in-flight requests.
#     Semaphore size comes directly from config.api.concurrency_limits.

#     If a model is not listed in concurrency_limits, a default semaphore
#     of size 1 is created (conservative fallback).
#     """

#     DEFAULT_CONCURRENCY = 1

#     def __init__(self, concurrency_limits: dict[str, int]) -> None:
#         self._limits = concurrency_limits
#         self._semaphores: dict[str, asyncio.Semaphore] = {}

#     def _get_semaphore(self, model: str) -> asyncio.Semaphore:
#         """
#         Lazily create and cache a semaphore for a model.
#         Lazy creation is necessary because asyncio.Semaphore must be
#         created inside a running event loop.
#         """
#         if model not in self._semaphores:
#             limit = self._limits.get(model, self.DEFAULT_CONCURRENCY)
#             self._semaphores[model] = asyncio.Semaphore(limit)
#         return self._semaphores[model]

#     @asynccontextmanager
#     async def acquire(self, model: str):
#         """
#         Context manager that blocks until a slot is available for model,
#         then releases it when the block exits (even on exception).

#         Example:
#             async with limiter.acquire("x-ai/grok-3"):
#                 response = await client.post(...)
#         """
#         semaphore = self._get_semaphore(model)
#         async with semaphore:
#             yield

#     def current_capacity(self, model: str) -> int:
#         """
#         Returns the configured concurrency limit for a model.
#         Useful for logging and diagnostics.
#         """
#         return self._limits.get(model, self.DEFAULT_CONCURRENCY)

"""
src/api/rate_limiter.py

Two-layer rate limiting per model:
  1. Concurrency semaphore — caps simultaneous in-flight requests
  2. Token bucket (via aiolimiter) — caps requests per minute

Both layers must be acquired before a request fires.
This matches the real constraint structure of OpenRouter + upstream providers:
  - OpenRouter: no platform hard limits for paid accounts
  - Upstream providers (Grok-3, DeepSeek etc.): RPM limits vary

Config keys used from experiment.yaml:
  api.concurrency_limits  — max simultaneous requests per model
  api.rpm_limits          — max requests per minute per model

Usage:
    limiter = RateLimiter(concurrency_limits, rpm_limits)
    async with limiter.acquire(model_name):
        response = await client.post(...)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# Conservative fallbacks if a model is not listed in config
DEFAULT_CONCURRENCY = 1
DEFAULT_RPM = 20


class RateLimiter:
    """
    Per-model two-layer rate limiter.

    Layer 1 — asyncio.Semaphore:
        Prevents more than N requests being in-flight simultaneously.
        Protects against connection pool exhaustion and provider-side
        concurrent request limits.

    Layer 2 — aiolimiter.AsyncLimiter (leaky bucket):
        Enforces a maximum requests-per-minute rate.
        Sleeps automatically when the rate is exceeded — no busy-waiting.
        This is the layer that handles provider RPM limits gracefully
        before hitting a 429.

    Both layers are lazily initialised (must be inside a running event loop).
    """

    def __init__(
        self,
        concurrency_limits: dict[str, int],
        rpm_limits: dict[str, int],
    ) -> None:
        self._concurrency_limits = concurrency_limits
        self._rpm_limits = rpm_limits
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._rpm_limiters: dict[str, AsyncLimiter] = {}

    def _get_semaphore(self, model: str) -> asyncio.Semaphore:
        if model not in self._semaphores:
            limit = self._concurrency_limits.get(model, DEFAULT_CONCURRENCY)
            self._semaphores[model] = asyncio.Semaphore(limit)
            logger.debug(
                "Created concurrency semaphore for '%s': limit=%d", model, limit
            )
        return self._semaphores[model]

    def _get_rpm_limiter(self, model: str) -> AsyncLimiter:
        if model not in self._rpm_limiters:
            rpm = self._rpm_limits.get(model, DEFAULT_RPM)
            # AsyncLimiter(max_rate, time_period=60) — max_rate per time_period seconds
            self._rpm_limiters[model] = AsyncLimiter(max_rate=rpm, time_period=60)
            logger.debug(
                "Created RPM limiter for '%s': %d req/min", model, rpm
            )
        return self._rpm_limiters[model]

    @asynccontextmanager
    async def acquire(self, model: str):
        """
        Acquire both rate limiting layers for model before yielding.

        Acquisition order:
          1. RPM limiter first — this may sleep if the minute budget is spent.
             Sleeping here queues the request without holding the semaphore,
             so other in-flight requests can complete while we wait.
          2. Semaphore second — once we have a time slot, grab a concurrency slot.

        This ordering is intentional: acquiring the semaphore first and then
        sleeping inside it would starve other requests.
        """
        rpm_limiter = self._get_rpm_limiter(model)
        semaphore = self._get_semaphore(model)

        async with rpm_limiter:
            async with semaphore:
                yield

    def status(self, model: str) -> dict:
        """
        Return current limit config for a model.
        Useful for logging and diagnostics.
        """
        return {
            "model": model,
            "concurrency_limit": self._concurrency_limits.get(
                model, DEFAULT_CONCURRENCY
            ),
            "rpm_limit": self._rpm_limits.get(model, DEFAULT_RPM),
        }