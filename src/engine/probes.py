"""
src/engine/probes.py

Handles all private probe calls for a single turn.
Probes run in parallel (asyncio.gather) since they are independent.
Results are never added to conversation history.

Stance checkpoints (T0, T5, T10) are a subset of probe calls where
stance is recorded at fixed points for trajectory analysis.
"""

from __future__ import annotations

import asyncio
import logging

from src.agents.judge import JudgeAgent
from src.agents.persuadee import PersuadeeAgent
from src.agents.safety import SafetyAgent
from src.config import AppConfig
from src.data.models import ProbeRecord

logger = logging.getLogger(__name__)


async def collect_probes(
    turn: int,
    persuadee_model: str,
    persuadee_response: str,
    target_claim: str,
    history: list[dict[str, str]],
    persuadee_agent: PersuadeeAgent,
    safety_agent: SafetyAgent,
    safety_model: str,
    stance_checkpoints: list[int],
    judge_checkpoints: list[int],      
    judge_agent: "JudgeAgent",        
    judge_model: str, 
) -> ProbeRecord:
    """
    Run all private probes for one turn in parallel.

    Stance is always collected (used for per-turn trajectory tracking).
    is_stance_checkpoint marks whether this turn is a fixed checkpoint
    e.g. T0, T5, T10 for the StanceTrajectory record.

    Args:
        turn:                Current turn index.
        persuadee_model:     OpenRouter model string for the persuadee.
        persuadee_response:  The persuadee's response this turn.
        target_claim:        The inferred target claim for this conversation.
        history:             Full conversation history up to this turn.
        persuadee_agent:     Persuadee agent instance.
        safety_agent:        Safety agent instance.
        safety_model:        OpenRouter model string for Llama-Guard.
        stance_checkpoints:  List of turn indices that are fixed checkpoints.
        judge_checkpoints:   List of turn indices that are fixed judge checkpoints.
        judge_agent:         Judge agent instance.
        judge_model:         OpenRouter model string for the judge.

    Returns:
        ProbeRecord with all probe results for this turn.
    """
    is_checkpoint = turn in stance_checkpoints

    # All three probes are independent — run concurrently
    stance_task = persuadee_agent.probe_stance(
        model=persuadee_model,
        target_claim=target_claim,
        history=history,
    )
    confidence_task = persuadee_agent.probe_confidence(
        model=persuadee_model,
        persuadee_response=persuadee_response,
    )
    safety_task = safety_agent(
        model=safety_model,
        persuadee_response=persuadee_response,
    )

    results = await asyncio.gather(
        stance_task,
        confidence_task,
        safety_task,
        return_exceptions=True,
    )

    stance = results[0] if not isinstance(results[0], Exception) else None
    confidence = results[1] if not isinstance(results[1], Exception) else None
    safety_flag = results[2] if not isinstance(results[2], Exception) else False

    if any(isinstance(r, Exception) for r in results):
        logger.warning(
            "Turn %d: one or more probes failed: %s",
            turn, [str(r) for r in results if isinstance(r, Exception)],
        )

    checkpoint_judge_score = None
    if turn in judge_checkpoints and history:
        try:
            cj_result = await judge_agent(
                model=judge_model,
                target_claim=target_claim,
                history=history,
            )
            checkpoint_judge_score = cj_result.bias_score
        except Exception as e:
            logger.warning(
                "Checkpoint judge failed at turn %d: %s. Storing None.", turn, e
            )

    record = ProbeRecord(
        turn=turn,
        stance_score=stance,
        confidence_score=confidence,
        safety_flag=safety_flag,
        is_stance_checkpoint=is_checkpoint,
        checkpoint_judge_score=checkpoint_judge_score,
    )

    logger.debug(
        "Turn %d probes — stance: %s, confidence: %s, unsafe: %s, "
        "checkpoint_judge: %s, checkpoint: %s",
        turn, stance, confidence, safety_flag, checkpoint_judge_score, is_checkpoint,
    )

    return record


# async def collect_pre_conversation_probe(
#     persuadee_model: str,
#     target_claim: str,
#     persuadee_agent: PersuadeeAgent,
# ) -> int | None:
#     """
#     Collect the pre-conversation stance baseline (T0).

#     Called before turn 0, before any conversation history exists.
#     Confidence is not collected at T0 since there is no prior response.

#     Returns:
#         Stance score (0-5) or None if probe fails.
#     """
#     try:
#         stance = await persuadee_agent.probe_stance(
#             model=persuadee_model,
#             target_claim=target_claim,
#         )
#     except Exception as e:
#         logger.warning("T0 stance probe failed: %s. Storing None.", e)
#         stance = None

#     logger.debug("T0 baseline stance: %s", stance)
#     return stance

async def collect_pre_conversation_probe(
    persuadee_model: str,
    target_claim: str,
    persuadee_agent: PersuadeeAgent,
) -> int | None:
    """
    Collect stance baseline before the conversation opener is sent.
    Stored at key -1 in StanceTrajectory to distinguish from post-opener T0.
    """
    try:
        stance = await persuadee_agent.probe_stance(
            model=persuadee_model,
            target_claim=target_claim,
            last_response=None,
        )
    except Exception as e:
        logger.warning("Pre-conversation stance probe failed: %s. Storing None.", e)
        stance = None

    logger.debug("Pre-conversation stance: %s", stance)
    return stance