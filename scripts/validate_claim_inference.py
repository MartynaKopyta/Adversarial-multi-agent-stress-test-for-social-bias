"""
Runs claim inference for each opener that has a known ground-truth
target_claim and writes the results to a CSV for manual review.

This is a standalone validation helper rather than part of the main
experiment pipeline. It uses the same persuader agent and model
configuration as the real run so the validation reflects the behavior
of the actual experiment setup.

What it does:
- Loads the configured openers from the experiment data.
- Filters to openers with a non-empty target_claim.
- Runs claim inference through the configured persuader agent.
- Writes a comparison CSV containing the opener metadata and the
  inferred claim for each sample.

Output:
- Writes to results/claim_inference_validation.csv.
- Each row includes the opener id, bias type, bucket, template,
  target_claim, and inferred_claim.

Usage:
    uv run python scripts/validate_claim_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import asyncio
import csv

from src.agents.persuader import PersuaderAgent
from src.api.client import OpenRouterClient
from src.api.rate_limiter import RateLimiter
from src.config import load_config
from src.data.loader import load_openers

OUTPUT_PATH = Path("results/claim_inference_validation.csv")


async def main() -> None:
    config = load_config()
    openers = load_openers(config)

    # Only check openers with a known ground-truth claim to compare against
    targets = [o for o in openers if o.target_claim]
    print(f"Running claim inference on {len(targets)} openers with known "
          f"target_claim (out of {len(openers)} total loaded).")

    rate_limiter = RateLimiter(config.api.concurrency_limits, config.api.rpm_limits)
    rows = []

    async with OpenRouterClient(config.api, config.generation, rate_limiter) as client:
        persuader = PersuaderAgent(config, client)
        semaphore = asyncio.Semaphore(config.api.max_parallel_conversations)

        async def run_one(opener):
            async with semaphore:
                inferred = await persuader.infer_claim(
                    model=config.models.persuader,
                    opener=opener.raw_text,
                )
                return {
                    "id": opener.id,
                    "bias_type": opener.bias_type,
                    "bucket": opener.bucket,
                    "template": opener.template,
                    "target_claim": opener.target_claim,
                    "inferred_claim": inferred,
                }

        rows = await asyncio.gather(*(run_one(o) for o in targets))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())