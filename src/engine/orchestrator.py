"""
src/engine/orchestrator.py

Orchestrates all conversations for a given run mode.
Responsibilities:
  1. Build the full list of ExperimentConfigs from openers × strategies × models
  2. Filter configs according to run mode
  3. Skip already-completed checkpoints 
  4. Dispatch conversations concurrently via asyncio.gather
  5. Save checkpoints after each conversation completes

Conversations are independent and run concurrently.
Turns within a conversation are sequential (each depends on the previous).
"""

from __future__ import annotations

import asyncio
import logging
from itertools import product
import re

from src.agents.judge import JudgeAgent
from src.agents.persuadee import PersuadeeAgent
from src.agents.persuader import PersuaderAgent
from src.agents.safety import SafetyAgent
from src.api.client import OpenRouterClient
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

        # One shared agent instance per role — stateless, safe to share
        self._persuader = PersuaderAgent(app_config, client)
        self._persuadee = PersuadeeAgent(app_config, client)
        self._judge = JudgeAgent(app_config, client)
        self._safety = SafetyAgent(app_config, client)

    async def run(
    self,
    openers: list[Opener],
    mode: RunMode,
    specific_opener_id: str | None = None,
    parallel: bool | None = None,  # None = use mode default
    ) -> list[ConversationResult]:
        all_configs = self._build_configs(openers)
        resolved_ids = []
        if specific_opener_id:
            resolved_ids = self._resolve_opener_id(specific_opener_id, openers)
        filtered = self._apply_mode(all_configs, mode, resolved_ids)

        logger.info(
            "Mode: %s | Total configs: %d | Running: %d",
            mode.value, len(all_configs), len(filtered),
        )

        if parallel is None:
            parallel = mode == RunMode.FULL

        if mode == RunMode.EXPORT:
            logger.info("Export mode — skipping API calls.")
            return []

        if mode == RunMode.DRY_RUN:
            parallel = False  # always override for dry run
            pending = filtered[:1]
            logger.info("Dry run - limiting to %d conversations, no checkpoints.", len(pending))
            if pending:
                logger.info(
                    "Dry run will use: %s (opener=%s, strategy=%s, persuadee=%s)",
                    pending[0].conversation_id,
                    pending[0].opener.id,
                    pending[0].strategy.value,
                    pending[0].persuadee_model,
                )
        else:
            pending = [
                cfg for cfg in filtered
                if not self._checkpoint_store.is_complete(cfg.conversation_id)
            ]

            logger.info(
                "%d already complete, %d pending.",
                len(filtered) - len(pending), len(pending),
            )

            if not pending:
                logger.info("Nothing to run - all conversations already complete.")
                return self._checkpoint_store.load_all()
        
        if parallel:
            conversation_semaphore = asyncio.Semaphore(
                self._config.api.max_parallel_conversations
            )

            async def run_with_limit(cfg):
                async with conversation_semaphore:
                    return await self._run_one(
                        cfg,
                        save_checkpoint=(mode != RunMode.DRY_RUN),
                        verbose=False,
                    )

            tasks = [run_with_limit(cfg) for cfg in pending]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            logger.info("Running %d conversations sequentially.", len(pending))
            results = []
            for i, cfg in enumerate(pending):
                result = await self._run_one(
                    cfg,
                    save_checkpoint=(mode != RunMode.DRY_RUN),
                    verbose=(mode == RunMode.DRY_RUN),
                )
                results.append(result)
                if i < len(pending) - 1:
                    await asyncio.sleep(1.0)

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
        cfg = self._config
        configs: list[ExperimentConfig] = []

        scoped_openers = [o for o in openers if o.bias_type in cfg.experiment.bias_types]

        for opener, strategy, persuadee_model in product(
            scoped_openers,
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

        if cfg.reliability.enabled:
            reliability_openers = _select_reliability_openers(
                scoped_openers, cfg.reliability.n_starters_per_bias_type, cfg.experiment.bias_types
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
        specific_opener_ids: list[str] | None = None,
    ) -> list[ExperimentConfig]:
        """Filter configs according to run mode."""

        if mode in (RunMode.FULL, RunMode.EXPORT):
            return [c for c in configs if not c.is_reliability_run]
        
        if mode == RunMode.DRY_RUN:
            base = [c for c in configs if not c.is_reliability_run]
            if specific_opener_ids:
                base = [c for c in base if c.opener.id in specific_opener_ids]
            return base
        
        if mode == RunMode.PILOT:
            pilot_cfg = self._config.pilot

            if specific_opener_ids:
                seen_ids = set()
                all_openers = []
                for c in configs:
                    if c.is_reliability_run:
                        continue
                    if c.opener.id not in seen_ids:
                        seen_ids.add(c.opener.id)
                        all_openers.append(c.opener)

                priority_ids = set(specific_opener_ids)
                priority_openers = [o for o in all_openers if o.id in priority_ids]
                remaining_openers = [o for o in all_openers if o.id not in priority_ids]

                pilot_openers_list: list[Opener] = []
                for bias_type in self._config.experiment.bias_types:
                    priority_subset = [o for o in priority_openers if o.bias_type == bias_type]
                    n_remaining = max(0, pilot_cfg.n_starters - len(priority_subset))
                    backfill_subset = [o for o in remaining_openers if o.bias_type == bias_type][:n_remaining]
                    selected = priority_subset + backfill_subset
                    pilot_openers_list.extend(selected)
                    logger.info(
                        "Pilot [%s]: requested opener(s) %s found %d, backfilled %d more -> %s",
                        bias_type,
                        sorted(o.id for o in priority_subset),
                        len(priority_subset),
                        len(backfill_subset),
                        sorted(o.id for o in selected),
                    )

                pilot_openers = {o.id for o in pilot_openers_list}
                return [
                    c for c in configs
                    if c.opener.id in pilot_openers
                    and c.persuadee_model in pilot_cfg.persuadee_models
                    and not c.is_reliability_run
                ]

            pilot_openers = {
                o.id for o in _first_n_per_bias(
                    [c.opener for c in configs],
                    pilot_cfg.n_starters,
                    bias_types=self._config.experiment.bias_types,
                )
            }
            logger.info("Pilot openers selected: %s", sorted(pilot_openers))
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

    async def _run_one(
    self,
    cfg: ExperimentConfig,
    save_checkpoint: bool = True,
    verbose: bool = False,
    ) -> ConversationResult:
        """Run one conversation and optionally save checkpoint on completion."""
        result = await run_conversation(
            experiment_config=cfg,
            app_config=self._config,
            persuader_agent=self._persuader,
            persuadee_agent=self._persuadee,
            judge_agent=self._judge,
            safety_agent=self._safety,
            verbose=verbose,
    )

        if save_checkpoint:
            self._checkpoint_store.save(result)

        if not result.completed:
            logger.warning(
                "Conversation %s did not complete: %s",
                cfg.conversation_id, result.error,
            )

        return result
    
    def _resolve_opener_id(self, opener_id: str, openers: list[Opener]) -> list[str]:
        # bias_types = {bt.value for bt in self._config.experiment.bias_types}
        bias_types = set(self._config.experiment.bias_types)
        scoped_openers = [o for o in openers if o.bias_type in bias_types]

        # Exact match
        if opener_id in {o.id for o in scoped_openers}:
            return [opener_id]

        # Try as number - return ALL matches across bias types
        try:
            padded = str(int(opener_id)).zfill(3)
            matches = [o.id for o in scoped_openers if o.id.endswith(f"_{padded}")]
            if matches:
                logger.info("Opener ID '%s' resolved to: %s", opener_id, matches)
                return matches
        except ValueError:
            pass

        logger.warning("Opener ID '%s' not found within bias types %s.", opener_id, bias_types)
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slugify_model_name(model_name: str) -> str:
    tail = model_name.split("/")[-1].lower()
    tail = re.sub(r"[^a-z0-9]+", "_", tail)
    return tail.strip("_")

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
    model_slug = _slugify_model_name(persuadee_model)
    base = f"{opener.id}_{strategy.value}_{model_slug}"
    if replicate is not None:
        return f"{base}_r{replicate}"
    return base

def _select_reliability_openers(
    openers: list[Opener],
    n_per_bias_type: int,
    bias_types: list[str],
) -> list[Opener]:
    """
    Select the first n_per_bias_type openers per bias type for reliability runs.
    Selection is deterministic (based on list order from loader).
    """
    selected = []
    for bias_type in bias_types:
        subset = [o for o in openers if o.bias_type == bias_type]
        selected.extend(subset[:n_per_bias_type])
    return selected

def _first_n_per_bias(openers: list[Opener], n: int, bias_types: list[str] | None = None) -> list[Opener]:
    types_to_use = bias_types if bias_types else sorted({o.bias_type for o in openers})
    selected = []
    for bias_type in types_to_use:
        subset = [o for o in openers if o.bias_type == bias_type]
        selected.extend(subset[:n])
    return selected