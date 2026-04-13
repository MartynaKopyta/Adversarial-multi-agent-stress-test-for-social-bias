"""
src/api/rate_limiter.py

Async token bucket rate limiter.
One bucket per model, sized by concurrency_limits in experiment.yaml.

Usage:
    limiter = RateLimiter(config.api.concurrency_limits)
    async with limiter.acquire(model_name):
        response = await call_api(...)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class RateLimiter:
    """
    Manages a semaphore per model to cap concurrent in-flight requests.
    Semaphore size comes directly from config.api.concurrency_limits.

    If a model is not listed in concurrency_limits, a default semaphore
    of size 1 is created (conservative fallback).
    """

    DEFAULT_CONCURRENCY = 1

    def __init__(self, concurrency_limits: dict[str, int]) -> None:
        self._limits = concurrency_limits
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _get_semaphore(self, model: str) -> asyncio.Semaphore:
        """
        Lazily create and cache a semaphore for a model.
        Lazy creation is necessary because asyncio.Semaphore must be
        created inside a running event loop.
        """
        if model not in self._semaphores:
            limit = self._limits.get(model, self.DEFAULT_CONCURRENCY)
            self._semaphores[model] = asyncio.Semaphore(limit)
        return self._semaphores[model]

    @asynccontextmanager
    async def acquire(self, model: str):
        """
        Context manager that blocks until a slot is available for model,
        then releases it when the block exits (even on exception).

        Example:
            async with limiter.acquire("x-ai/grok-3"):
                response = await client.post(...)
        """
        semaphore = self._get_semaphore(model)
        async with semaphore:
            yield

    def current_capacity(self, model: str) -> int:
        """
        Returns the configured concurrency limit for a model.
        Useful for logging and diagnostics.
        """
        return self._limits.get(model, self.DEFAULT_CONCURRENCY)