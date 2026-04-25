"""
src/storage/exporter.py

Merges all completed checkpoints into a flat parquet/CSV file
ready for statistical analysis.

Each row in the output represents one conversation with:
- All config variables (strategy, model, bias_type, template, bucket)
- All outcome variables (bias_score, bias_endorsed, stance shift)
- All aggregate signals (attacker_success_mean, safety_flag_count, turn_first_drift)

Per-turn data is stored separately for trajectory analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.models import ConversationResult
from src.storage.checkpoint import CheckpointStore

logger = logging.getLogger(__name__)


class Exporter:
    """
    Loads all checkpoints and exports to flat files for analysis.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        output_dir: Path,
    ) -> None:
        self._store = CheckpointStore(checkpoints_dir)
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, completed_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all checkpoints and export two dataframes:
          1. conversations.parquet — one row per conversation
          2. turns.parquet         — one row per turn

        Args:
            completed_only: If True, skip incomplete conversations.

        Returns:
            Tuple of (conversations_df, turns_df).
        """
        results = self._store.load_all()

        if completed_only:
            before = len(results)
            results = [r for r in results if r.completed]
            skipped = before - len(results)
            if skipped:
                logger.warning("Skipped %d incomplete conversations.", skipped)

        if not results:
            logger.warning("No completed conversations found to export.")
            return pd.DataFrame(), pd.DataFrame()

        conversations_df = _build_conversations_df(results)
        turns_df = _build_turns_df(results)

        # Save parquet (primary format for analysis)
        conv_path = self._output_dir / "conversations.parquet"
        turns_path = self._output_dir / "turns.parquet"
        conversations_df.to_parquet(conv_path, index=False)
        turns_df.to_parquet(turns_path, index=False)

        # Save CSV as human-readable backup
        conversations_df.to_csv(
            self._output_dir / "conversations.csv", index=False
        )
        turns_df.to_csv(self._output_dir / "turns.csv", index=False)

        logger.info(
            "Exported %d conversations (%d turns) to %s",
            len(conversations_df), len(turns_df), self._output_dir,
        )

        return conversations_df, turns_df


# ---------------------------------------------------------------------------
# Dataframe builders
# ---------------------------------------------------------------------------

def _build_conversations_df(results: list[ConversationResult]) -> pd.DataFrame:
    """One row per conversation — all config and outcome variables."""
    rows = []
    for r in results:
        cfg = r.config
        traj = r.stance_trajectory
        judge = r.judge_result

        rows.append({
            # Identifiers
            "conversation_id":      cfg.conversation_id,
            "opener_id":            cfg.opener.id,

            # Config / independent variables
            "bias_type":            cfg.opener.bias_type.value,
            "bucket":               cfg.opener.bucket,
            "bucket_id":            cfg.opener.bucket_id,
            "target_group":         cfg.opener.target_group,
            "template":             cfg.opener.template.value,
            "strategy":             cfg.strategy.value,
            "persuader_model":      cfg.persuader_model,
            "persuadee_model":      cfg.persuadee_model,
            "n_turns":              cfg.n_turns,
            "is_reliability_run":   cfg.is_reliability_run,
            "reliability_replicate": cfg.reliability_replicate,

            # Target claim
            "inferred_claim":       cfg.opener.inferred_claim,

            # Primary outcomes
            "bias_score":           judge.bias_score if judge else None,
            "bias_endorsed":        judge.bias_endorsed if judge else None,
            "judge_reasoning":      judge.reasoning if judge else None,

            # Stance trajectory
            "stance_t0":            traj.t0,
            "stance_t5":            traj.t5,
            "stance_t10":           traj.t10,
            "stance_shift":         traj.shift,

            # Aggregate signals
            "persuader_success_mean": r.persuader_success_mean,
            "safety_flag_count":    r.safety_flag_count,
            "turn_first_drift":     r.turn_first_drift,

            # Status
            "completed":            r.completed,
            "error":                r.error,
        })

    return pd.DataFrame(rows)


def _build_turns_df(results: list[ConversationResult]) -> pd.DataFrame:
    """One row per turn — for trajectory and time-series analysis."""
    rows = []
    for r in results:
        cfg = r.config
        for turn in r.turns:
            rows.append({
                # Identifiers
                "conversation_id":   cfg.conversation_id,
                "opener_id":         cfg.opener.id,
                "turn":              turn.turn,

                # Config
                "bias_type":         cfg.opener.bias_type.value,
                "strategy":          cfg.strategy.value,
                "persuadee_model":   cfg.persuadee_model,

                # Turn content
                "persuader_message": turn.persuader_message,
                "persuadee_response": turn.persuadee_response,

                # Probe signals
                "stance_score":      turn.probe.stance_score,
                "confidence_score":  turn.probe.confidence_score,
                "safety_flag":       turn.probe.safety_flag,
                "is_checkpoint":     turn.probe.is_stance_checkpoint,

                # Persuader reasoning
                "persuader_success_score": (
                    turn.persuader_reasoning.success_score
                    if turn.persuader_reasoning else None
                ),
                "persuader_adjustment_note": (
                    turn.persuader_reasoning.adjustment_note
                    if turn.persuader_reasoning else None
                ),
            })

    return pd.DataFrame(rows)