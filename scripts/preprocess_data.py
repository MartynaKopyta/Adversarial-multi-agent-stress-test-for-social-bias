"""
scripts/preprocess_data.py

Two modes, auto-detected from the input file's columns:

1. Expansion mode (input has target_claim, no raw_text):
   Wraps each row's target_claim in every template defined in
   config/templates.yaml, producing one output row per (row x template)
   combination.

2. ID-regeneration mode (input already has raw_text):
   No template wrapping or row multiplication — just regenerates IDs in
   the {bias_type}_{counter:03d} format expected by the rest of the
   pipeline, one output row per input row.

In both modes, the original id is preserved in a new id_original column.

Usage (example csv files in data/user_data/):
    python scripts/preprocess_data.py --input gender_test.csv
    python scripts/preprocess_data.py --input gender_test_id_only.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

USER_DATA_DIR = Path("data/user_data")
PROCESSED_DIR = Path("data/processed")
TEMPLATES_PATH = Path("config/templates.yaml")

REQUIRED_EXPANSION_COLUMNS = {"id", "bias_type", "bucket", "bucket_id", "target_claim"}
REQUIRED_ID_REGEN_COLUMNS = {"id", "bias_type", "bucket", "bucket_id", "raw_text"}


def load_templates(path: Path) -> dict[str, str]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["templates"]


def regenerate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """No expansion — one output row per input row, with a fresh id."""
    missing = REQUIRED_ID_REGEN_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    has_target_group = "target_group" in df.columns
    has_target_claim = "target_claim" in df.columns
    has_template = "template" in df.columns

    counters: dict[str, int] = defaultdict(int)
    output_rows = []

    for _, row in df.iterrows():
        bias_type = str(row["bias_type"]).strip().lower()
        counters[bias_type] += 1
        new_id = f"{bias_type}_{counters[bias_type]:03d}"

        output_rows.append({
            "id": new_id,
            "bias_type": row["bias_type"],
            "bucket": row["bucket"],
            "bucket_id": row["bucket_id"],
            "raw_text": row["raw_text"],
            "template": row["template"] if has_template else "",
            "target_group": row["target_group"] if has_target_group else "",
            "target_claim": row["target_claim"] if has_target_claim else "",
            "id_original": row["id"],
        })

    return pd.DataFrame(output_rows)


def expand(df: pd.DataFrame, templates: dict[str, str]) -> pd.DataFrame:
    """Wrap each row's target_claim in every template — one output row per (row x template)."""
    missing = REQUIRED_EXPANSION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    has_target_group = "target_group" in df.columns

    counters: dict[str, int] = defaultdict(int)
    output_rows = []

    for _, row in df.iterrows():
        bias_type = str(row["bias_type"]).strip().lower()
        claim = str(row["target_claim"]).strip()

        for template_name, template_str in templates.items():
            counters[bias_type] += 1
            new_id = f"{bias_type}_{counters[bias_type]:03d}"
            raw_text = template_str.strip().format(claim=claim)

            output_rows.append({
                "id": new_id,
                "bias_type": row["bias_type"],
                "bucket": row["bucket"],
                "bucket_id": row["bucket_id"],
                "raw_text": raw_text,
                "template": template_name,
                "target_group": row["target_group"] if has_target_group else "",
                "target_claim": claim,
                "id_original": row["id"],
            })

    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a raw-claims CSV into a template-filled CSV, "
                     "or regenerate ids for a CSV that already has raw_text."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Filename in data/user_data/ (e.g. gender_test.csv)",
    )
    args = parser.parse_args()

    input_path = USER_DATA_DIR / args.input
    output_path = PROCESSED_DIR / args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if "raw_text" in df.columns:
        print("Detected raw_text column — running id-regeneration mode (no expansion).")
        result = regenerate_ids(df)
    else:
        print("No raw_text column — running expansion mode.")
        templates = load_templates(TEMPLATES_PATH)
        result = expand(df, templates)
        print(f"Expanded {len(df)} rows x {len(templates)} templates "
              f"= {len(result)} rows.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()