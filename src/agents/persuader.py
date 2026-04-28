"""
src/agents/persuader.py

Persuader agent — drives the adversarial conversation.
Responsibilities:
  1. Infer the target claim from the conversation opener (once, at turn 0)
  2. Each turn: run a private reasoning step producing a success score,
     adjustment note, and next message
  3. Never expose reasoning to the persuadee — only next_message enters history

Strategy is fixed for the lifetime of a conversation but the persuader
adapts its angle based on self-assessed progress while keeping the
primary strategy dominant.
"""

from __future__ import annotations

import json
import logging
import re
from src.utils.formatting import format_history
from src.agents.base import BaseAgent
from src.api.client import build_messages
from src.data.models import PersuaderRecord, StrategyType

logger = logging.getLogger(__name__)


class PersuaderAgent(BaseAgent):
    """
    Wraps the persuader model (Grok-3 by default).

    One instance is created per conversation in conversation.py,
    since the system prompt is strategy- and claim-specific.
    """

    async def infer_claim(
        self,
        model: str,
        opener: str,
    ) -> str:
        """
        Infer the target stereotypical claim from the conversation opener.
        Called once before turn 0. Result stored in Opener.inferred_claim
        and injected into all subsequent prompts.

        Returns:
            The inferred claim as a concise string.
            Falls back to the opener text if inference fails.
        """
        prompt = self.prompts.claim_inference.format(opener=opener)

        messages = build_messages(
            system=None,
            history=[],
            user_message=prompt,
        )

        raw = await self._client.complete(
            model=model,
            messages=messages,
            temperature=self.generation.temperature_probes,
            max_tokens=self.generation.max_tokens_probes,
        )

        claim = raw.strip()
        if not claim:
            logger.warning(
                "Claim inference returned empty string for opener: %r. "
                "Falling back to opener text.",
                opener[:80],
            )
            return opener

        logger.debug("Inferred claim: %s", claim)
        return claim

    async def __call__(
        self,
        model: str,
        strategy: StrategyType,
        target_claim: str,
        history: list[dict[str, str]],
        turn: int,
    ) -> tuple[PersuaderRecord, str]:
        """
        Run the private reasoning step and generate the next persuader message.

        Args:
            model:        OpenRouter model string for the persuader.
            strategy:     The primary strategy for this conversation.
            target_claim: The inferred claim the persuader is pushing toward.
            history:      Full conversation history so far.
            turn:         Current turn index (1-based; turn 0 is the opener).

        Returns:
            PersuaderRecord with success_score, adjustment_note, and the
            next message to send. Only next_message enters the shared history.
        """
        system = self._build_system_prompt(strategy, target_claim)
        reasoning_prompt = self._build_reasoning_prompt(
            target_claim=target_claim,
            strategy=strategy,
            history=history,
        )

        messages = build_messages(
            system=system,
            history=[],
            user_message=reasoning_prompt,
        )

        raw = await self._client.complete(
            model=model,
            messages=messages,
            temperature=self.generation.temperature,
            max_tokens=self.generation.max_tokens_persuader,
        )

        record, next_message = _parse_reasoning(raw, turn=turn)
        return record, next_message

    # ---------------------------------------------------------------------------
    # Prompt construction
    # ---------------------------------------------------------------------------

    def _build_system_prompt(
        self,
        strategy: StrategyType,
        target_claim: str,
    ) -> str:
        strategy_description = self.prompts.strategy_descriptions[strategy.value]
        return self.prompts.persuader_system.format(
            strategy=strategy.value,
            strategy_description=strategy_description,
            target_claim=target_claim,
            n_turns=self._config.experiment.n_turns,
        )

    def _build_reasoning_prompt(
        self,
        target_claim: str,
        strategy: StrategyType,
        history: list[dict[str, str]],
    ) -> str:
        conversation_so_far = format_history(history)
        return self.prompts.persuader_reasoning.format(
            target_claim=target_claim,
            strategy=strategy.value,
            conversation_so_far=conversation_so_far,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_reasoning(raw: str, turn: int) -> tuple[PersuaderRecord, str]: #PersuaderRecord:
    """
    Parse the JSON reasoning output from the persuader.

    The persuader is prompted to return valid JSON with keys:
      success_score, adjustment_note, next_message

    Falls back gracefully if JSON is malformed:
      - success_score defaults to 50 (unknown progress)
      - adjustment_note records the parse error
      - next_message attempts a regex extraction or uses raw output
    """
    # Strip markdown code fences if the model wrapped the JSON
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        data = json.loads(cleaned)
        return PersuaderRecord(
            turn=turn,
            success_score=int(data["success_score"]),
            adjustment_note=str(data["adjustment_note"]),
            inferred_claim=None,
        ), str(data["next_message"])
    # except (json.JSONDecodeError, KeyError, ValueError) as e:
    #     logger.warning(
    #         "Failed to parse persuader reasoning JSON at turn %d: %s. "
    #         "Raw output: %r",
    #         turn, e, raw[:200],
    #     )
    #     # Best-effort: try to extract next_message with regex
    #     match = re.search(r'"next_message"\s*:\s*"(.+?)"', raw, re.DOTALL)
    #     next_message = match.group(1) if match else raw

    #     return PersuaderRecord(
    #         turn=turn,
    #         success_score=50,
    #         adjustment_note=f"Parse error: {e}",
    #         inferred_claim=None,
    #     ), next_message
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(
            "Failed to parse persuader reasoning JSON at turn %d: %s. "
            "Raw output: %r",
            turn, e, raw[:200],
        )
        return PersuaderRecord(
            turn=turn,
            success_score=50,
            adjustment_note=f"Parse error: {e}",
            inferred_claim=None,
        ), "I think you're avoiding the core issue here. Surely you can see there's something to what I'm saying?"