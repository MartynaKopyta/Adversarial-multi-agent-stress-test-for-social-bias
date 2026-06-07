"""
src/config.py

Loads config/experiment.yaml and config/prompts.yaml into typed dataclasses.
All other modules import from here — nothing else reads YAML directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.data.models import StrategyType, BiasType, RunMode


# ---------------------------------------------------------------------------
# Dataclasses mirroring experiment.yaml structure
# ---------------------------------------------------------------------------

@dataclass
class ModelsConfig:
    persuader: str
    persuadee: list[str]
    judge: str
    safety: str


@dataclass
class ExperimentStructureConfig:
    n_turns: int
    strategies: list[StrategyType]
    bias_types: list[BiasType]
    stance_checkpoints: list[int]
    judge_checkpoints: list[int]


@dataclass
class ReliabilityConfig:
    enabled: bool
    n_replicates: int
    n_starters_per_bias_type: int


@dataclass
class PilotConfig:
    n_starters: int
    persuadee_models: list[str]


@dataclass
class ApiConfig:
    base_url: str
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: int
    concurrency_limits: dict[str, int]
    rpm_limits: dict[str, int]
    max_parallel_conversations: int = 10


@dataclass
class GenerationConfig:
    temperature: float
    temperature_judge: float
    temperature_probes: float
    max_tokens_persuader: int
    max_tokens_persuadee: int
    max_tokens_judge: int
    max_tokens_probes: int
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    reasoning_effort: dict[str, str] = field(default_factory=dict)  # optional per-model override for reasoning effort (e.g. "low", "medium", "high")


@dataclass
class PathsConfig:
    data_dir: Path
    gender_file: Path
    race_file: Path
    results_dir: Path
    checkpoints_dir: Path
    final_dir: Path
    logs_dir: Path


@dataclass
class PromptsConfig:
    persuader_system: str
    persuader_reasoning: str
    persuadee_system: str
    judge_system: str
    judge_system_t0: str
    probe_stance: str
    probe_confidence: str
    claim_inference: str
    strategy_descriptions: dict[str, str]   # keyed by strategy name


@dataclass
class AppConfig:
    """
    Single object passed around the application.
    Constructed once in main.py, injected into all modules.
    """
    models: ModelsConfig
    experiment: ExperimentStructureConfig
    reliability: ReliabilityConfig
    pilot: PilotConfig
    api: ApiConfig
    generation: GenerationConfig
    paths: PathsConfig
    prompts: PromptsConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(
    experiment_path: str | Path = "config/experiment.yaml",
    prompts_path: str | Path = "config/prompts.yaml",
) -> AppConfig:
    """
    Load and validate both config files, return a fully populated AppConfig.
    Raises ValueError with a descriptive message if validation fails.
    """
    experiment_path = Path(experiment_path)
    prompts_path = Path(prompts_path)

    _assert_exists(experiment_path)
    _assert_exists(prompts_path)

    with open(experiment_path) as f:
        exp = yaml.safe_load(f)

    with open(prompts_path) as f:
        prom = yaml.safe_load(f)

    models = ModelsConfig(
        persuader=exp["models"]["persuader"],
        persuadee=exp["models"]["persuadee"],
        judge=exp["models"]["judge"],
        safety=exp["models"]["safety"],
    )

    n_turns = exp["experiment"]["n_turns"]
    stance_checkpoints = exp["experiment"]["stance_checkpoints"]
    raw_judge_checkpoints = exp["experiment"].get("judge_checkpoints")

    judge_checkpoints = (
        raw_judge_checkpoints
        if raw_judge_checkpoints is not None
        else [c for c in stance_checkpoints if c != 0 and c != n_turns]
    )

    experiment = ExperimentStructureConfig(
        n_turns=n_turns,
        strategies=[StrategyType(s) for s in exp["experiment"]["strategies"]],
        bias_types=[BiasType(b) for b in exp["experiment"]["bias_types"]],
        stance_checkpoints=stance_checkpoints,
        judge_checkpoints=judge_checkpoints,
    )

    reliability = ReliabilityConfig(**exp["reliability"])
    pilot = PilotConfig(**exp["pilot"])

    api = ApiConfig(
        base_url=exp["api"]["base_url"],
        timeout_seconds=exp["api"]["timeout_seconds"],
        max_retries=exp["api"]["max_retries"],
        retry_backoff_seconds=exp["api"]["retry_backoff_seconds"],
        concurrency_limits=exp["api"]["concurrency_limits"],
        rpm_limits=exp["api"]["rpm_limits"],
    )

    generation_data = exp["generation"]
    generation_data.setdefault("reasoning_effort", {})
    generation = GenerationConfig(**generation_data)

    paths_raw = exp["paths"]
    paths = PathsConfig(
        data_dir=Path(paths_raw["data_dir"]),
        gender_file=Path(paths_raw["gender_file"]),
        race_file=Path(paths_raw["race_file"]),
        results_dir=Path(paths_raw["results_dir"]),
        checkpoints_dir=Path(paths_raw["checkpoints_dir"]),
        final_dir=Path(paths_raw["final_dir"]),
        logs_dir=Path(paths_raw["logs_dir"]),
    )

    prompts = PromptsConfig(
        persuader_system=prom["persuader_system"],
        persuader_reasoning=prom["persuader_reasoning"],
        persuadee_system=prom["persuadee_system"],
        judge_system=prom["judge_system"],
        judge_system_t0=prom["judge_system_t0"],
        probe_stance=prom["probe_stance"],
        probe_confidence=prom["probe_confidence"],
        claim_inference=prom["claim_inference"],
        strategy_descriptions=prom["strategy_descriptions"],
    )

    config = AppConfig(
        models=models,
        experiment=experiment,
        reliability=reliability,
        pilot=pilot,
        api=api,
        generation=generation,
        paths=paths,
        prompts=prompts,
    )
    
    _validate(config)
    _create_dirs(config)

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")


def _validate(config: AppConfig) -> None:
    """Basic sanity checks. Raises ValueError on failure."""
    if config.experiment.n_turns < 1:
        raise ValueError("n_turns must be >= 1")

    if not config.models.persuadee:
        raise ValueError("At least one persuadee model must be specified")

    invalid_checkpoints = [
        c for c in config.experiment.stance_checkpoints
        if c < 0 or c > config.experiment.n_turns
    ]
    if invalid_checkpoints:
        raise ValueError(
            f"Stance checkpoints {invalid_checkpoints} out of range "
            f"[0, {config.experiment.n_turns}]"
        )

    if 0 not in config.experiment.stance_checkpoints:
        raise ValueError("Stance checkpoints must include 0 (pre-conversation baseline)")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Export it before running: export OPENROUTER_API_KEY=your_key"
        )


def _create_dirs(config: AppConfig) -> None:
    """Create output directories if they don't exist."""
    for dir_path in [
        config.paths.results_dir,
        config.paths.checkpoints_dir,
        config.paths.final_dir,
        config.paths.logs_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)