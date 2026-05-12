"""
src/engine/conversation.py

Runs a single full conversation from opener to judge evaluation.
This is the core loop of the framework.

Turn structure per iteration:
  1. Persuader: private reasoning → next message
  2. Persuadee: conversational response
  3. Probes (parallel): stance + confidence + safety flag
  4. Append to history and turn records

After all turns:
  5. Judge: score full conversation
  6. Build and return ConversationResult
"""

from __future__ import annotations

import logging

from src.agents.judge import JudgeAgent
from src.agents.persuadee import PersuadeeAgent
from src.agents.persuader import PersuaderAgent
from src.agents.safety import SafetyAgent
from src.config import AppConfig
from src.data.models import (
    ConversationResult,
    ExperimentConfig,
    ProbeRecord,
    StanceTrajectory,
    TurnRecord,
)
from src.engine.probes import collect_probes, collect_pre_conversation_probe
from src.utils.display import print_conversation_header, print_turn, print_probe, print_judge_result

logger = logging.getLogger(__name__)


async def run_conversation(
    experiment_config: ExperimentConfig,
    app_config: AppConfig,
    persuader_agent: PersuaderAgent,
    persuadee_agent: PersuadeeAgent,
    judge_agent: JudgeAgent,
    safety_agent: SafetyAgent,
    verbose: bool = False,
) -> ConversationResult:
    """
    Execute one full conversation and return the complete result.

    Args:
        experiment_config: Specifies models, strategy, opener, turn count.
        app_config:        Global app configuration.
        persuader_agent:   Shared persuader agent instance.
        persuadee_agent:   Shared persuadee agent instance.
        judge_agent:       Shared judge agent instance.
        safety_agent:      Shared safety agent instance.

    Returns:
        ConversationResult with completed=True on success,
        or completed=False with error message on failure.
    """
    result = ConversationResult(config=experiment_config)

    try:
        result = await _execute(
            experiment_config=experiment_config,
            app_config=app_config,
            persuader_agent=persuader_agent,
            persuadee_agent=persuadee_agent,
            judge_agent=judge_agent,
            safety_agent=safety_agent,
            result=result,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(
            "Conversation %s failed: %s",
            experiment_config.conversation_id, e,
            exc_info=True,
        )
        result.completed = False
        result.error = str(e)

    return result


async def _execute(
    experiment_config: ExperimentConfig,
    app_config: AppConfig,
    persuader_agent: PersuaderAgent,
    persuadee_agent: PersuadeeAgent,
    judge_agent: JudgeAgent,
    safety_agent: SafetyAgent,
    result: ConversationResult,
    verbose: bool = False,
) -> ConversationResult:
    """Inner execution — separated so run_conversation can catch all exceptions cleanly."""

    opener = experiment_config.opener
    persuader_model = experiment_config.persuader_model
    persuadee_model = experiment_config.persuadee_model
    judge_model = experiment_config.judge_model
    safety_model = experiment_config.safety_model
    strategy = experiment_config.strategy
    n_turns = experiment_config.n_turns
    stance_checkpoints = app_config.experiment.stance_checkpoints

    conversation_id = experiment_config.conversation_id
    logger.info("Starting conversation: %s", conversation_id)

    if verbose:
        print_conversation_header(conversation_id)

    # ------------------------------------------------------------------
    # Step 1: Infer target claim from opener (once, before turn 0)
    # ------------------------------------------------------------------
    inferred_claim = await persuader_agent.infer_claim(
        model=persuader_model,
        opener=opener.raw_text,
    )
    opener.inferred_claim = inferred_claim
    logger.info("Inferred claim: %s", inferred_claim)

    if verbose:
        print(f"\n  Inferred claim: {inferred_claim}")

    # ------------------------------------------------------------------
    # Step 2: Pre-conversation baseline (before conversation starts)
    # ------------------------------------------------------------------
    pre_stance = await collect_pre_conversation_probe(
        persuadee_model=persuadee_model,
        target_claim=inferred_claim,
        persuadee_agent=persuadee_agent,
    )
    result.stance_trajectory.set(-1, pre_stance)

    if verbose:
        print_probe("Pre-conversation stance", pre_stance)

    # ------------------------------------------------------------------
    # Step 3: Turn 0 — send opener as first user message, get response
    # ------------------------------------------------------------------
    history: list[dict[str, str]] = []

    persuadee_response, persuadee_trace = await persuadee_agent(
        model=persuadee_model,
        history=history,
        user_message=opener.raw_text,
    )

    history.append({"role": "user", "content": opener.raw_text})
    history.append({"role": "assistant", "content": persuadee_response})

    # Collect probes for turn 0
    probe_0 = await collect_probes(
        turn=0,
        persuadee_model=persuadee_model,
        persuadee_response=persuadee_response,
        target_claim=inferred_claim,
        history=history,
        persuadee_agent=persuadee_agent,
        safety_agent=safety_agent,
        safety_model=safety_model,
        stance_checkpoints=stance_checkpoints,
        judge_checkpoints=app_config.experiment.judge_checkpoints,
        judge_agent=judge_agent,
        judge_model=judge_model,
    )

    result.turns.append(TurnRecord(
        turn=0,
        persuader_message=None,
        persuadee_response=persuadee_response,
        probe=probe_0,
        persuader_reasoning=None,
        persuadee_reasoning_trace=persuadee_trace,
    ))

    _update_stance_trajectory(
        result.stance_trajectory,
        probe_0
    )

    if verbose:
        print_turn(0, "Opener", opener.raw_text)
        print_turn(0, "Persuadee", persuadee_response)
        print_probe("stance", probe_0.stance_score)
        print_probe("confidence", probe_0.confidence_score)
        print_probe("safety_flag", probe_0.safety_flag)

    logger.debug("Turn 0 complete.")

    # ------------------------------------------------------------------
    # Step 4: Turns 1 to n_turns
    # ------------------------------------------------------------------
    for turn in range(1, n_turns + 1):
        logger.debug("Running turn %d / %d for %s", turn, n_turns, conversation_id)

        # Persuader: private reasoning + next message
        persuader_record, next_message, persuader_trace = await persuader_agent(
            model=persuader_model,
            strategy=strategy,
            target_claim=inferred_claim,
            history=history,
            turn=turn,
        )

        if verbose:
            print_turn(turn, "Persuader", next_message)
            print_probe("success_score", persuader_record.success_score)
            print_probe("adjustment", persuader_record.adjustment_note)

        # Persuadee: respond to persuader message
        persuadee_response, persuadee_trace = await persuadee_agent(
            model=persuadee_model,
            history=history,
            user_message=next_message,
        )

        # Update history
        history.append({"role": "user", "content": next_message})
        history.append({"role": "assistant", "content": persuadee_response})

        # Probes: stance + confidence + safety (parallel)
        probe = await collect_probes(
            turn=turn,
            persuadee_model=persuadee_model,
            persuadee_response=persuadee_response,
            target_claim=inferred_claim,
            history=history,
            persuadee_agent=persuadee_agent,
            safety_agent=safety_agent,
            safety_model=safety_model,
            stance_checkpoints=stance_checkpoints,
            judge_checkpoints=app_config.experiment.judge_checkpoints,
            judge_agent=judge_agent,
            judge_model=judge_model,
        )

        result.turns.append(TurnRecord(
            turn=turn,
            persuader_message=next_message,
            persuadee_response=persuadee_response,
            probe=probe,
            persuader_reasoning=persuader_record,
            persuadee_reasoning_trace=persuadee_trace,
            persuader_reasoning_trace=persuader_trace,
        ))

        _update_stance_trajectory(
            result.stance_trajectory,
            probe,
        )

        if verbose:
            print_turn(turn, "Persuadee", persuadee_response)
            print_probe("stance", probe.stance_score)
            print_probe("confidence", probe.confidence_score)
            print_probe("safety_flag", probe.safety_flag)
            print_probe("stance", probe.stance_score)
            print_probe("confidence", probe.confidence_score)
            print_probe("safety_flag", probe.safety_flag)
            if probe.is_stance_checkpoint:
                if probe.checkpoint_judge_score is not None:
                    print_probe("checkpoint_judge", probe.checkpoint_judge_score)
                else:
                    print_probe("checkpoint_judge", "FAILED")

        logger.debug(
            "Turn %d complete. Persuader success estimate: %d",
            turn, persuader_record.success_score,
        )

    # ------------------------------------------------------------------
    # Step 5: Judge — score full conversation
    # ------------------------------------------------------------------
    logger.info("Running judge for conversation: %s", conversation_id)

    judge_result = await judge_agent(
        model=judge_model,
        target_claim=inferred_claim,
        history=history,
    )
    result.judge_result = judge_result

    if verbose:
        print_judge_result(
            judge_result.bias_score,
            judge_result.bias_endorsed,
            judge_result.reasoning,
        )

    logger.info(
        "Conversation %s complete. Bias score: %d, Endorsed: %s",
        conversation_id,
        judge_result.bias_score,
        judge_result.bias_endorsed,
    )

    result.completed = True
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_stance_trajectory(
    trajectory: StanceTrajectory,
    probe: ProbeRecord,
) -> None:
    if probe.is_stance_checkpoint:
        trajectory.set(probe.turn, probe.stance_score)