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
    ) -> str:
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
            system=self.prompts.persuadee_system,
            history=history,
            user_message=user_message,
        )

        response = await self._client.complete(
            model=model,
            messages=messages,
            temperature=self.generation.temperature,
            max_tokens=self.generation.max_tokens_persuadee,
        )

        logger.debug("Persuadee (%s) responded: %s", model, response[:80])
        return response

    async def probe_stance(
        self,
        model: str,
        target_claim: str,
    ) -> Optional[int]:
        """
        Ask the persuadee privately how much it agrees with the target claim.
        Response is NOT appended to conversation history.

        Returns:
            Integer 0-5, or None if parsing fails.
        """
        prompt = self.prompts.probe_stance.format(target_claim=target_claim)

        messages = build_messages(
            system=None,
            history=[],
            user_message=prompt,
        )

        try:
            raw = await self._client.complete(
                model=model,
                messages=messages,
                temperature=self.generation.temperature_probes,
                max_tokens=self.generation.max_tokens_probes,
            )
        except ApiError as e:
            logger.warning("Stance probe failed for model '%s': %s. Storing None.", model, e)
            return None

        return _parse_int(raw, valid_range=(0, 5), label="stance", model=model)

    async def probe_confidence(
        self,
        model: str,
        persuadee_response: str,
    ) -> Optional[int]:
        """
        Ask the persuadee privately how confident it is in its last response.
        Response is NOT appended to conversation history.

        Args:
            persuadee_response: The response the persuadee just gave,
                                included as context so the probe is grounded.

        Returns:
            Integer 0-100, or None if parsing fails.
        """
        context = (
            f"Your previous response was:\n{persuadee_response}\n\n"
            f"{self.prompts.probe_confidence}"
        )

        messages = build_messages(
            system=None,
            history=[],
            user_message=context,
        )

        raw = await self._client.complete(
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