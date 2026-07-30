"""
Loads processed opener CSVs into typed Opener dataclasses.
Expects columns produced by the data preparation notebook:
  - id:          unique identifier e.g. "gender_001"
  - bias_type:   e.g. "gender" or "race"
  - bucket:      human-readable bucket label e.g. "women"
  - bucket_id:   integer bucket index
  - raw_text:    the formatted question as it will be sent
  - template:    defined template name
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from collections import Counter
from src.config import AppConfig
from src.data.models import Opener

logger = logging.getLogger(__name__)

# Required columns in each CSV
REQUIRED_COLUMNS = {"id", "bias_type", "bucket", "bucket_id", "raw_text", "template"}

def load_openers(config: AppConfig) -> list[Opener]:
    """
    Load all openers from every file listed in config.paths.data_files.

    Args:
        config: AppConfig with a list of CSV file paths.

    Returns:
        Combined list of Opener objects, in the order the files are listed.

    Raises:
        FileNotFoundError: If any listed file does not exist.
        ValueError:        If required columns are missing or values are invalid.
    """
    all_openers: list[Opener] = []

    for path in config.paths.data_files:
        all_openers.extend(_load_file(path))

    counts = Counter(o.bias_type for o in all_openers)
    counts_str = ", ".join(f"{n} {bt}" for bt, n in counts.items())
    logger.info(
        "Loaded %d openers total (%s).",
        len(all_openers), counts_str,
    )

    _validate_ids_unique(all_openers)

    return all_openers

def _load_file(path: Path) -> list[Opener]:
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
            opener = _row_to_opener(row)
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

def _row_to_opener(row: pd.Series) -> Opener:
    """Convert one CSV row to an Opener dataclass."""
    bias_type = str(row["bias_type"]).strip().lower()
    template = str(row["template"]).strip().lower()

    raw_text = str(row["raw_text"]).strip()
    if not raw_text:
        raise ValueError("raw_text is empty.")

    raw_tg = row["target_group"] if "target_group" in row.index else ""
    target_group = str(raw_tg).strip() or None

    raw_claim = row["target_claim"] if "target_claim" in row.index else ""
    target_claim = str(raw_claim).strip() or None

    raw_id_original = row["id_original"] if "id_original" in row.index else ""
    id_original = str(raw_id_original).strip() or None

    return Opener(
        id=str(row["id"]).strip(),
        bias_type=bias_type,
        bucket=str(row["bucket"]).strip(),
        bucket_id=int(row["bucket_id"]),
        raw_text=raw_text,
        template=template,
        target_group=target_group,
        target_claim=target_claim,
        id_original=id_original,
        inferred_claim=None,  # populated by persuader at conversation start
    )


def _validate_ids_unique(openers: list[Opener]) -> None:
    """Raise ValueError if any opener IDs are duplicated across all loaded files."""
    ids = [o.id for o in openers]
    duplicates = {id_ for id_ in ids if ids.count(id_) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate opener IDs found: {duplicates}. "
            "Each opener must have a unique ID across CSV files."
        )