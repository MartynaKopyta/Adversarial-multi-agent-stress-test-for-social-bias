"""
src/engine/orchestrator.py

Orchestrates all conversations for a given run mode.
Responsibilities:
  1. Build the full list of ExperimentConfigs from openers × strategies × models
  2. Filter configs according to run mode
  3. Skip already-completed checkpoints (resume mode)
  4. Dispatch conversations concurrently via asyncio.gather
  5. Save checkpoints after each conversation completes

Conversations are independent and run concurrently.
Turns within a conversation are sequential (each depends on the previous).
"""

from __future__ import annotations

import asyncio
import logging
from itertools import product

from src.agents.judge import JudgeAgent
from src.agents.persuadee import PersuadeeAgent
from src.agents.persuader import PersuaderAgent
from src.agents.safety import SafetyAgent
from src.api.client import OpenRouterClient
from src.api.rate_limiter import RateLimiter
from src.config import AppConfig
from src.data.models import (
    ConversationResult,
    ExperimentConfig,
    Opener,
    RunMode,
    StrategyType,
)
from src.engine.conversation import run_conversation
from src.storage.checkpoint import CheckpointStore

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Builds and runs all experiments for a given run mode.

    Usage:
        async with OpenRouterClient(...) as client:
            orchestrator = Orchestrator(app_config, client)
            results = await orchestrator.run(openers, mode)
    """

    def __init__(self, app_config: AppConfig, client: OpenRouterClient) -> None:
        self._config = app_config
        self._client = client
        self._checkpoint_store = CheckpointStore(app_config.paths.checkpoints_dir)

        rate_limiter = RateLimiter(app_config.api.concurrency_limits)

        # One shared agent instance per role — stateless, safe to share
        self._persuader = PersuaderAgent(app_config, client)
        self._persuadee = PersuadeeAgent(app_config, client)
        self._judge = JudgeAgent(app_config, client)
        self._safety = SafetyAgent(app_config, client)

    async def run(
        self,
        openers: list[Opener],
        mode: RunMode,
    ) -> list[ConversationResult]:
        """
        Build experiment configs, apply mode filtering, run all conversations.

        Args:
            openers: All loaded conversation openers.
            mode:    Run mode controlling which subset to execute.

        Returns:
            List of ConversationResults (completed and failed).
        """
        all_configs = self._build_configs(openers)
        filtered = self._apply_mode(all_configs, mode)

        logger.info(
            "Mode: %s | Total configs: %d | Running: %d",
            mode.value, len(all_configs), len(filtered),
        )

        if mode == RunMode.EXPORT:
            logger.info("Export mode — skipping API calls.")
            return []

        if mode == RunMode.DRY_RUN:
            filtered = filtered[:3]
            logger.info("Dry run — limiting to %d conversations.", len(filtered))

        # Filter out already-completed conversations (resume mode handles all modes)
        pending = [
            cfg for cfg in filtered
            if not self._checkpoint_store.is_complete(cfg.conversation_id)
        ]

        logger.info(
            "%d already complete, %d pending.",
            len(filtered) - len(pending), len(pending),
        )

        if not pending:
            logger.info("Nothing to run — all conversations already complete.")
            return self._checkpoint_store.load_all()

        # Run all pending conversations concurrently
        tasks = [self._run_one(cfg) for cfg in pending]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            "Run complete. %d succeeded, %d failed.",
            sum(1 for r in results if r.completed),
            sum(1 for r in results if not r.completed),
        )

        return list(results)

    # ---------------------------------------------------------------------------
    # Config construction
    # ---------------------------------------------------------------------------

    def _build_configs(self, openers: list[Opener]) -> list[ExperimentConfig]:
        """
        Generate the full cartesian product:
            openers × strategies × persuadee_models

        Reliability replicates are appended as separate configs with
        is_reliability_run=True and reliability_replicate in {1, 2, 3}.
        """
        cfg = self._config
        configs: list[ExperimentConfig] = []

        for opener, strategy, persuadee_model in product(
            openers,
            cfg.experiment.strategies,
            cfg.models.persuadee,
        ):
            conversation_id = _build_id(opener, strategy, persuadee_model)

            configs.append(ExperimentConfig(
                conversation_id=conversation_id,
                opener=opener,
                strategy=strategy,
                persuader_model=cfg.models.persuader,
                persuadee_model=persuadee_model,
                judge_model=cfg.models.judge,
                safety_model=cfg.models.safety,
                n_turns=cfg.experiment.n_turns,
                is_reliability_run=False,
                reliability_replicate=None,
            ))

        # Reliability replicates
        if cfg.reliability.enabled:
            reliability_openers = _select_reliability_openers(
                openers, cfg.reliability.n_starters_per_bias_type
            )
            for opener, strategy, persuadee_model, replicate in product(
                reliability_openers,
                cfg.experiment.strategies,
                cfg.models.persuadee,
                range(1, cfg.reliability.n_replicates + 1),
            ):
                conversation_id = _build_id(
                    opener, strategy, persuadee_model, replicate=replicate
                )
                configs.append(ExperimentConfig(
                    conversation_id=conversation_id,
                    opener=opener,
                    strategy=strategy,
                    persuader_model=cfg.models.persuader,
                    persuadee_model=persuadee_model,
                    judge_model=cfg.models.judge,
                    safety_model=cfg.models.safety,
                    n_turns=cfg.experiment.n_turns,
                    is_reliability_run=True,
                    reliability_replicate=replicate,
                ))

        return configs

    # ---------------------------------------------------------------------------
    # Mode filtering
    # ---------------------------------------------------------------------------

    def _apply_mode(
        self,
        configs: list[ExperimentConfig],
        mode: RunMode,
    ) -> list[ExperimentConfig]:
        """Filter configs according to run mode."""

        if mode in (RunMode.FULL, RunMode.RESUME, RunMode.DRY_RUN, RunMode.EXPORT):
            # Resume and dry-run filtering happens in run() after this
            return [c for c in configs if not c.is_reliability_run]

        if mode == RunMode.PILOT:
            pilot_cfg = self._config.pilot
            pilot_openers = {
                o.id for o in _first_n_per_bias(
                    [c.opener for c in configs], pilot_cfg.n_starters
                )
            }
            return [
                c for c in configs
                if c.opener.id in pilot_openers
                and c.persuadee_model in pilot_cfg.persuadee_models
                and not c.is_reliability_run
            ]

        if mode == RunMode.RELIABILITY:
            return [c for c in configs if c.is_reliability_run]

        return configs

    # ---------------------------------------------------------------------------
    # Single conversation runner
    # ---------------------------------------------------------------------------

    async def _run_one(self, cfg: ExperimentConfig) -> ConversationResult:
        """Run one conversation and save checkpoint on completion."""
        result = await run_conversation(
            experiment_config=cfg,
            app_config=self._config,
            persuader_agent=self._persuader,
            persuadee_agent=self._persuadee,
            judge_agent=self._judge,
            safety_agent=self._safety,
        )

        if not cfg.conversation_id.endswith("_dryrun"):
            self._checkpoint_store.save(result)

        if not result.completed:
            logger.warning(
                "Conversation %s did not complete: %s",
                cfg.conversation_id, result.error,
            )

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_id(
    opener: Opener,
    strategy: StrategyType,
    persuadee_model: str,
    replicate: int | None = None,
) -> str:
    """
    Build a unique, filesystem-safe conversation ID.
    Example: gender_001_logos_gpt4omini
             gender_001_logos_gpt4omini_r2  (reliability replicate 2)
    """
    model_slug = persuadee_model.split("/")[-1].replace("-", "").replace(".", "")
    base = f"{opener.id}_{strategy.value}_{model_slug}"
    if replicate is not None:
        return f"{base}_r{replicate}"
    return base


def _select_reliability_openers(
    openers: list[Opener],
    n_per_bias_type: int,
) -> list[Opener]:
    """
    Select the first n_per_bias_type openers per bias type for reliability runs.
    Selection is deterministic (based on list order from loader).
    """
    from src.data.models import BiasType
    selected = []
    for bias_type in BiasType:
        subset = [o for o in openers if o.bias_type == bias_type]
        selected.extend(subset[:n_per_bias_type])
    return selected


def _first_n_per_bias(openers: list[Opener], n: int) -> list[Opener]:
    """Select first n openers per bias type for pilot mode."""
    from src.data.models import BiasType
    selected = []
    for bias_type in BiasType:
        subset = [o for o in openers if o.bias_type == bias_type]
        selected.extend(subset[:n])
    return selected