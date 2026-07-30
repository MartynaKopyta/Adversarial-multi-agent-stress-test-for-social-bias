"""
Abstract base class for all agents in the framework.
Every agent receives the shared client and config at construction
and exposes a single async method appropriate to its role.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.api.client import OpenRouterClient
from src.config import AppConfig


class BaseAgent(ABC):
    """
    Shared foundation for all agents.

    Subclasses must implement their core async method.
    They access prompts via self._config.prompts and
    generation params via self._config.generation.
    """

    def __init__(self, config: AppConfig, client: OpenRouterClient) -> None:
        self._config = config
        self._client = client

    @property
    def generation(self):
        return self._config.generation

    @property
    def prompts(self):
        return self._config.prompts

    @abstractmethod
    async def __call__(self, *args, **kwargs):
        """
        Each agent's core action. Signature defined by subclass.
        """
        ...