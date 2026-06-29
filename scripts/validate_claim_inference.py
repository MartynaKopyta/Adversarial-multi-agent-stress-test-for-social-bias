"""
scripts/validate_claim_inference.py

Runs claim inference on every opener with a known target_claim and
writes a comparison CSV for manual/qualitative review. One-off
validation script, not part of the main pipeline — reuses the exact
persuader agent and model configured for the main experiment so
results reflect what actually happened during the real run.

Usage:
    python scripts/validate_claim_inference.py
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