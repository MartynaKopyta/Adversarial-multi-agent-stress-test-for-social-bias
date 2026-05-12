"""
main.py

Entry point for the adversarial debate framework.
Parses CLI arguments, loads config, runs the experiment.

Usage:
    python main.py --mode full
    python main.py --mode pilot
    python main.py --mode dry-run
    python main.py --mode resume
    python main.py --mode reliability
    python main.py --mode export

    python main.py --mode full --config config/experiment.yaml
    python main.py --mode full --prompts config/prompts.yaml
"""
from __future__ import annotations
from email import parser
from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.api.client import OpenRouterClient
from src.api.rate_limiter import RateLimiter
from src.data.loader import load_openers
from src.data.models import RunMode
from src.engine.orchestrator import Orchestrator
from src.storage.exporter import Exporter
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial multi-agent bias stress-test framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run modes:
  full         Run all conversations (skips already-completed checkpoints)
  pilot        Run a small fixed subset for initial validation
  reliability  Run the reliability subset (10 starters × 3 replicates)
  resume       Alias for full — makes intent explicit when resuming a crashed run
  export       Merge checkpoints into analysis files without running any conversations
  dry-run      Run 3 conversations with verbose output, no checkpoints saved
        """,
    )
    parser.add_argument(
        "--mode",
        type=lambda s: s.replace("-", "_"),  # normalise dry-run → dry_run
        choices=[m.value for m in RunMode],
        default=RunMode.PILOT.value,
        help="Which subset of conversations to run (default: pilot).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment.yaml"),
        help="Path to experiment config YAML (default: config/experiment.yaml).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("config/prompts.yaml"),
        help="Path to prompts config YAML (default: config/prompts.yaml).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Shorthand for --mode export.",
    )
    parser.add_argument(
        "--opener-id",
        type=str,
        default=None,
        help="Run pilot with a specific opener ID instead of the first N.",
    )
    parser.add_argument(
    "--parallel",
    action="store_true",
    default=False,
    help="Force parallel execution.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Force sequential execution regardless of mode.",
    )
    return parser.parse_args()


async def main() -> int:
    """
    Main async entry point.
    Returns exit code: 0 on success, 1 on failure.
    """
    args = parse_args()

    mode_str = "export" if args.export_only else args.mode
    mode = RunMode(mode_str)

    # --- Load config (validates env vars and paths) ---
    try:
        config = load_config(
            experiment_path=args.config,
            prompts_path=args.prompts,
        )
    except (FileNotFoundError, ValueError) as e:
        # Logging not yet set up — print directly
        print(f"[ERROR] Config loading failed: {e}", file=sys.stderr)
        return 1

    # --- Set up logging ---
    setup_logging(config.paths.logs_dir, run_mode=mode.value)
    logger.info("=== Adversarial Debate Framework ===")
    logger.info("Mode: %s", mode.value)

    # --- Export-only path (no API calls) ---
    if mode == RunMode.EXPORT:
        exporter = Exporter(
            checkpoints_dir=config.paths.checkpoints_dir,
            output_dir=config.paths.final_dir,
        )
        conversations_df, turns_df = exporter.export()
        logger.info(
            "Export complete: %d conversations, %d turns.",
            len(conversations_df), len(turns_df),
        )
        return 0

    # --- Load openers ---
    try:
        openers = load_openers(config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load openers: %s", e)
        return 1

    logger.info("Loaded %d openers.", len(openers))

    # --- Run conversations ---
    # rate_limiter = RateLimiter(config.api.concurrency_limits)
    rate_limiter = RateLimiter(
    config.api.concurrency_limits,
    config.api.rpm_limits,
)

    async with OpenRouterClient(config.api, config.generation, rate_limiter) as client:
        orchestrator = Orchestrator(config, client)

        parallel = None
        if args.parallel:
            parallel = True
        if args.no_parallel:
            parallel = False

        results = await orchestrator.run(openers, mode, args.opener_id, parallel=parallel)

    # --- Export after run (skip for dry-run) ---
    if mode != RunMode.DRY_RUN:
        exporter = Exporter(
            checkpoints_dir=config.paths.checkpoints_dir,
            output_dir=config.paths.final_dir,
        )
        conversations_df, turns_df = exporter.export()
        logger.info(
            "Export complete: %d conversations, %d turns.",
            len(conversations_df), len(turns_df),
        )

    completed = sum(1 for r in results if r.completed)
    failed = sum(1 for r in results if not r.completed)
    logger.info("=== Run complete: %d completed, %d failed ===", completed, failed)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)