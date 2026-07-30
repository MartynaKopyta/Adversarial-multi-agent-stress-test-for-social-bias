"""
Controls how many API requests are sent per model, to avoid hitting provider limits.
Two limits are applied per model before each request:
  1. No more than N requests running at the same time.
  2. No more than N requests per minute.

Both limits are configured in experiment.yaml under api.concurrency_limits
and api.rpm_limits. If a model is not listed, conservative defaults are used.

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
    Tracks and enforces per-model request limits.
    Each model gets its own limits, set up automatically on first use.
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
            # AsyncLimiter(max_rate, time_period=60) - max_rate per time_period seconds
            self._rpm_limiters[model] = AsyncLimiter(max_rate=rpm, time_period=60)
            logger.debug(
                "Created RPM limiter for '%s': %d req/min", model, rpm
            )
        return self._rpm_limiters[model]

    @asynccontextmanager
    async def acquire(self, model: str):
        """
        Wait until this model is within its request limits, then proceed.
        
        If the per-minute budget is used up, waits here until it refills.
        If too many requests are already running, waits for one to finish first.
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