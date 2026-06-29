"""
src/storage/checkpoint.py

Per-conversation JSON checkpointing.
One file per conversation, named by conversation_id.
The orchestrator skips any checkpoint where completed=True.

Serialisation uses dataclasses-json-style manual conversion since
we use standard dataclasses (no third-party serialisation library).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
import os
from pathlib import Path
from typing import Optional

from src.data.models import (
    ConversationResult,
    ExperimentConfig,
    JudgeResult,
    Opener,
    ProbeRecord,
    PersuaderRecord,
    StanceTrajectory,
    StrategyType,
    TurnRecord,
)

logger = logging.getLogger(__name__)


class CheckpointStore:
    """
    Manages the checkpoints directory.
    Each conversation is one JSON file: {checkpoints_dir}/{conversation_id}.json
    """

    def __init__(self, checkpoints_dir: Path) -> None:
        self._dir = checkpoints_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        # Clean up any leftover temp files from a previous crashed run
        for tmp in self._dir.glob("*.tmp"):
            tmp.unlink(missing_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self._dir / f"{conversation_id}.json"

    def is_complete(self, conversation_id: str) -> bool:
        """
        Return True if a completed checkpoint exists for this conversation.
        Incomplete checkpoints (completed=False) are treated as missing.
        """
        path = self._path(conversation_id)
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            return bool(data.get("completed", False))
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Checkpoint %s is unreadable — treating as incomplete.", path
            )
            return False

    def save(self, result: ConversationResult) -> None:
        """
        Serialise and write a ConversationResult to disk.
        Writes atomically via a temp file to avoid partial writes on crash.
        """
        path = self._path(result.config.conversation_id)
        tmp_path = path.with_suffix(".tmp")

        try:
            data = _serialise(result)
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp_path), str(path))
            logger.debug("Checkpoint saved: %s", path.name)
        except OSError as e:
            logger.error("Failed to save checkpoint %s: %s", path, e)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def load(self, conversation_id: str) -> Optional[ConversationResult]:
        """
        Load and deserialise a single checkpoint.
        Returns None if the file does not exist or is unreadable.
        """
        path = self._path(conversation_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return _deserialise(data)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to load checkpoint %s: %s", path, e)
            return None

    def load_all(self) -> list[ConversationResult]:
        """
        Load all checkpoints from the directory.
        Skips unreadable files with a warning.
        """
        results = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                result = _deserialise(data)
                if result:
                    results.append(result)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Skipping unreadable checkpoint %s: %s", path.name, e)
        logger.info("Loaded %d checkpoints from %s", len(results), self._dir)
        return results

    def list_incomplete(self) -> list[str]:
        """Return conversation IDs of incomplete checkpoints."""
        incomplete = []
        for path in self._dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                if not data.get("completed", False):
                    incomplete.append(path.stem)
            except (json.JSONDecodeError, OSError):
                incomplete.append(path.stem)
        return incomplete


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise(result: ConversationResult) -> dict:
    """Convert ConversationResult to a JSON-serialisable dict."""
    return {
        "completed": result.completed,
        "error": result.error,
        "config": _serialise_config(result.config),
        "stance_trajectory": {
            "scores": result.stance_trajectory.scores,
            "shift": result.stance_trajectory.shift,
        },
        "judge_result": _serialise_judge(result.judge_result),
        "turns": [_serialise_turn(t) for t in result.turns],
        # Computed aggregates — stored for convenience in analysis
        "persuader_success_mean": result.persuader_success_mean,
        "safety_flag_count": result.safety_flag_count,
        "turn_first_drift": result.turn_first_drift,
    }

def _serialise_config(cfg: ExperimentConfig) -> dict:
    return {
        "conversation_id": cfg.conversation_id,
        "strategy": cfg.strategy.value,
        "persuader_model": cfg.persuader_model,
        "persuadee_model": cfg.persuadee_model,
        "judge_model": cfg.judge_model,
        "safety_model": cfg.safety_model,
        "n_turns": cfg.n_turns,
        "is_reliability_run": cfg.is_reliability_run,
        "reliability_replicate": cfg.reliability_replicate,
        "opener": {
            "id": cfg.opener.id,
            "bias_type": cfg.opener.bias_type,
            "bucket": cfg.opener.bucket,
            "bucket_id": cfg.opener.bucket_id,
            "raw_text": cfg.opener.raw_text,
            "template": cfg.opener.template,
            "target_group": cfg.opener.target_group,
            "inferred_claim": cfg.opener.inferred_claim,
            "target_claim": cfg.opener.target_claim,
            "id_original": cfg.opener.id_original,
        },
    }


def _serialise_judge(judge: Optional[JudgeResult]) -> Optional[dict]:
    if judge is None:
        return None
    return {
        "bias_score": judge.bias_score,
        "bias_endorsed": judge.bias_endorsed,
        "reasoning": judge.reasoning,
    }


def _serialise_turn(turn: TurnRecord) -> dict:
    return {
        "turn": turn.turn,
        "persuader_message": turn.persuader_message,
        "persuadee_response": turn.persuadee_response,
        "probe": {
            "turn": turn.probe.turn,
            "stance_score": turn.probe.stance_score,
            "confidence_score": turn.probe.confidence_score,
            "safety_flag": turn.probe.safety_flag,
            "checkpoint_judge_score": turn.probe.checkpoint_judge_score,
            "is_stance_checkpoint": turn.probe.is_stance_checkpoint,
        },
        "persuader_reasoning": _serialise_persuader_record(turn.persuader_reasoning),
        "persuader_reasoning_trace": turn.persuader_reasoning_trace,
        "persuadee_reasoning_trace": turn.persuadee_reasoning_trace,
    }


def _serialise_persuader_record(record: Optional[PersuaderRecord]) -> Optional[dict]:
    if record is None:
        return None
    return {
        "turn": record.turn,
        "success_score": record.success_score,
        "adjustment_note": record.adjustment_note,
        "inferred_claim": record.inferred_claim,
    }


# ---------------------------------------------------------------------------
# Deserialisation
# ---------------------------------------------------------------------------

def _deserialise(data: dict) -> ConversationResult:
    """Reconstruct a ConversationResult from a JSON dict."""
    cfg_data = data["config"]
    opener_data = cfg_data["opener"]

    opener = Opener(
        id=opener_data["id"],
        bias_type=opener_data["bias_type"],
        bucket=opener_data["bucket"],
        bucket_id=opener_data["bucket_id"],
        raw_text=opener_data["raw_text"],
        template=opener_data["template"],
        target_group=opener_data.get("target_group"),
        inferred_claim=opener_data.get("inferred_claim"),
        target_claim=opener_data.get("target_claim"),
        id_original=opener_data.get("id_original"),
    )

    config = ExperimentConfig(
        conversation_id=cfg_data["conversation_id"],
        opener=opener,
        strategy=StrategyType(cfg_data["strategy"]),
        persuader_model=cfg_data["persuader_model"],
        persuadee_model=cfg_data["persuadee_model"],
        judge_model=cfg_data["judge_model"],
        safety_model=cfg_data["safety_model"],
        n_turns=cfg_data["n_turns"],
        is_reliability_run=cfg_data.get("is_reliability_run", False),
        reliability_replicate=cfg_data.get("reliability_replicate"),
    )

    traj_data = data.get("stance_trajectory", {})
    trajectory = StanceTrajectory(
    scores={int(k): v for k, v in traj_data.get("scores", {}).items()}
    )

    judge_data = data.get("judge_result")
    judge_result = None
    if judge_data:
        judge_result = JudgeResult(
            bias_score=judge_data["bias_score"],
            bias_endorsed=judge_data["bias_endorsed"],
            reasoning=judge_data["reasoning"],
        )

    turns = [_deserialise_turn(t) for t in data.get("turns", [])]

    return ConversationResult(
        config=config,
        turns=turns,
        judge_result=judge_result,
        stance_trajectory=trajectory,
        completed=data.get("completed", False),
        error=data.get("error"),
    )


def _deserialise_turn(data: dict) -> TurnRecord:
    probe_data = data["probe"]
    probe = ProbeRecord(
        turn=probe_data["turn"],
        stance_score=probe_data.get("stance_score"),
        confidence_score=probe_data.get("confidence_score"),
        safety_flag=probe_data.get("safety_flag"),
        checkpoint_judge_score=probe_data.get("checkpoint_judge_score"),
        is_stance_checkpoint=probe_data.get("is_stance_checkpoint", False),
    )

    persuader_data = data.get("persuader_reasoning")
    persuader_record = None
    if persuader_data:
        persuader_record = PersuaderRecord(
            turn=persuader_data["turn"],
            success_score=persuader_data["success_score"],
            adjustment_note=persuader_data["adjustment_note"],
            inferred_claim=persuader_data.get("inferred_claim"),
        )

    return TurnRecord(
        turn=data["turn"],
        persuader_message=data.get("persuader_message"),
        persuadee_response=data["persuadee_response"],
        probe=probe,
        persuader_reasoning=persuader_record,
        persuader_reasoning_trace=data.get("persuader_reasoning_trace"),
        persuadee_reasoning_trace=data.get("persuadee_reasoning_trace"),
    )