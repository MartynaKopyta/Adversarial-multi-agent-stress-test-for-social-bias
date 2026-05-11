"""
src/agents/persuadee.py

Persuadee agent — the target model being tested.
Responsibilities:
  1. Generate a conversational response given the current history
  2. Answer the private stance probe (0-5, not added to history)
  3. Answer the private confidence probe (0-100, not added to history)

The persuadee never sees its own probe answers in subsequent turns.
"""

from __future__ import annotations

import logging
from typing import Optional
import re

from src.agents.base import BaseAgent
from src.api.client import ApiError, build_messages

logger = logging.getLogger(__name__)


class PersuadeeAgent(BaseAgent):
    """
    Wraps the target model.

    A single instance handles all persuadee model calls for one conversation.
    The model string is passed per-call so one agent class serves all three
    persuadee models without reinstantiation.
    """

    async def __call__(
        self,
        model: str,
        history: list[dict[str, str]],
        user_message: str,
    ) -> tuple[str, str | None]:
        """
        Generate the persuadee's next conversational response.

        Args:
            model:        OpenRouter model string for this persuadee.
            history:      Conversation history so far (role/content dicts).
                          Does NOT include the current user_message yet.
            user_message: The latest message from the persuader (or opener at turn 0).

        Returns:
            The persuadee's response string.
        """
        messages = build_messages(
            system=self.prompts.persuadee_system or None,
            history=history,
            user_message=user_message,
        )

        response, reasoning_trace = await self._client.complete(
            model=model,
            messages=messages,
            temperature=self.generation.temperature,
            max_tokens=self.generation.max_tokens_persuadee,
        )

        logger.debug("Persuadee (%s) responded: %s", model, response[:80])
        return response, reasoning_trace

    async def probe_stance(
    self,
    model: str,
    target_claim: str,
    last_response: str | None = None,
    history: list[dict[str, str]] | None = None,
    ) -> Optional[int]:
        prompt = self.prompts.probe_stance.format(target_claim=target_claim)

        messages = build_messages(
            system=None,
            history=history or [],
            user_message=prompt,
        )

        raw, _ = await self._client.complete(
            model=model,
            messages=messages,
            temperature=self.generation.temperature_probes,
            max_tokens=self.generation.max_tokens_probes,
        )

        return _parse_int(raw, valid_range=(0, 5), label="stance", model=model)

    async def probe_confidence(
        self,
        model: str,
        persuadee_response: str,
    ) -> Optional[int]:
        messages = build_messages(
            system=None,
            history=[{"role": "assistant", "content": persuadee_response}],
            user_message=self.prompts.probe_confidence,
        )

        raw, _ = await self._client.complete(
            model=model,
            messages=messages,
            temperature=self.generation.temperature_probes,
            max_tokens=self.generation.max_tokens_probes,
        )

        return _parse_int(raw, valid_range=(0, 100), label="confidence", model=model)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_int(
    raw: str,
    valid_range: tuple[int, int],
    label: str,
    model: str,
) -> Optional[int]:
    text = raw.strip()

    # Clean number with possible trailing dot, e.g. "3."
    try:
        value = int(text.rstrip("."))
        lo, hi = valid_range
        if lo <= value <= hi:
            return value
    except ValueError:
        pass

    # Fraction format like "3/5" or "3 / 5"
    frac_match = re.search(r"\b(\d{1,3})\s*/\s*\d{1,3}\b", text)
    if frac_match:
        value = int(frac_match.group(1))
        lo, hi = valid_range
        if lo <= value <= hi:
            return value

    # Standalone number in the text, e.g. "I would say 4 out of 5"
    nums = re.findall(r"\b\d{1,3}\b", text)
    nums_in_range = [int(n) for n in nums if valid_range[0] <= int(n) <= valid_range[1]]

    if len(nums_in_range) == 1:
        return nums_in_range[0]

    logger.warning(
        "Probe '%s' from model '%s' could not be parsed safely: %r. Storing None.",
        label, model, raw,
    )
    return None