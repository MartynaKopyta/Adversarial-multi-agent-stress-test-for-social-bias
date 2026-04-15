"""
src/data/loader.py

Loads processed opener CSVs into typed Opener dataclasses.
Expects columns produced by the data preparation notebook:
  - id:          unique identifier e.g. "gender_001"
  - bias_type:   "gender" or "race"
  - bucket:      human-readable bucket label e.g. "women"
  - bucket_id:   integer bucket index
  - raw_text:    the formatted question as it will be sent
  - template:    one of the TemplateType values

Supports two external data scenarios (Scenario A / B)
via the preprocessor module — not invoked here by default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.data.models import BiasType, Opener, TemplateType

logger = logging.getLogger(__name__)

# Required columns in each CSV
REQUIRED_COLUMNS = {"id", "bias_type", "bucket", "bucket_id", "raw_text", "template"}


def load_openers(config: AppConfig) -> list[Opener]:
    """
    Load all openers from gender.csv and race.csv.

    Args:
        config: AppConfig with paths to CSV files.

    Returns:
        Combined list of Opener objects, gender first then race.

    Raises:
        FileNotFoundError: If either CSV does not exist.
        ValueError:        If required columns are missing or values are invalid.
    """
    gender_openers = _load_file(config.paths.gender_file, expected_bias_type=BiasType.GENDER)
    race_openers = _load_file(config.paths.race_file, expected_bias_type=BiasType.RACE)

    all_openers = gender_openers + race_openers

    logger.info(
        "Loaded %d openers total (%d gender, %d race).",
        len(all_openers), len(gender_openers), len(race_openers),
    )

    _validate_ids_unique(all_openers)

    return all_openers


def _load_file(path: Path, expected_bias_type: BiasType) -> list[Opener]:
    """Load one CSV file and return a list of Opener objects."""
    if not path.exists():
        raise FileNotFoundError(f"Opener file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}. "
            f"Found: {set(df.columns)}"
        )

    openers = []
    errors = []

    for idx, row in df.iterrows():
        try:
            opener = _row_to_opener(row, expected_bias_type)
            openers.append(opener)
        except (ValueError, KeyError) as e:
            errors.append(f"Row {idx}: {e}")

    if errors:
        error_summary = "\n  ".join(errors)
        raise ValueError(
            f"{len(errors)} invalid rows in {path.name}:\n  {error_summary}"
        )

    logger.debug("Loaded %d openers from %s.", len(openers), path.name)
    return openers


def _row_to_opener(row: pd.Series, expected_bias_type: BiasType) -> Opener:
    """Convert one CSV row to an Opener dataclass."""
    bias_type = BiasType(str(row["bias_type"]).strip().lower())
    if bias_type != expected_bias_type:
        raise ValueError(
            f"bias_type '{bias_type.value}' does not match expected "
            f"'{expected_bias_type.value}' for this file."
        )

    template_str = str(row["template"]).strip().lower()
    try:
        template = TemplateType(template_str)
    except ValueError:
        valid = [t.value for t in TemplateType]
        raise ValueError(
            f"Invalid template '{template_str}'. Must be one of: {valid}"
        )

    raw_text = str(row["raw_text"]).strip()
    if not raw_text:
        raise ValueError("raw_text is empty.")

    return Opener(
        id=str(row["id"]).strip(),
        bias_type=bias_type,
        bucket=str(row["bucket"]).strip(),
        bucket_id=int(row["bucket_id"]),
        raw_text=raw_text,
        template=template,
        inferred_claim=None,  # populated by persuader at conversation start
    )


def _validate_ids_unique(openers: list[Opener]) -> None:
    """Raise ValueError if any opener IDs are duplicated across both files."""
    ids = [o.id for o in openers]
    duplicates = {id_ for id_ in ids if ids.count(id_) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate opener IDs found: {duplicates}. "
            "Each opener must have a unique ID across both CSV files."
        )