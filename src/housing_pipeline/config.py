"""Paths and settings for the housing affordability pipeline.

Every path used anywhere in the package resolves from here, so a checkout in a
different location -- or a test run against a temp directory -- needs no edits
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: this file is src/housing_pipeline/config.py
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

# Raw upstream downloads. Gitignored -- rebuilt on demand by the cache layer.
CACHE_DIR = DATA_DIR / "cache"

# Legacy raw files that were committed before the pipeline could fetch them.
# Sources use these as a fallback when a remote fetch is unavailable.
RAW_DIR = DATA_DIR

# Built outputs (Parquet).
BUILD_DIR = DATA_DIR / "build"

PANEL_PATH = BUILD_DIR / "panel.parquet"
FEATURES_PATH = BUILD_DIR / "features.parquet"

# Identify ourselves to public data hosts. BLS in particular rejects requests
# with no User-Agent, which is what made its bulk files look unreachable.
USER_AGENT = "housing-affordability-pipeline/1.0 (research; contact via repo)"

# Affordability threshold: a metro is "unaffordable" once median home value
# exceeds this multiple of median household income. Chosen in W07 by comparing
# onset dates at 4.0/4.5/5.0/5.5 -- 5.0 gave the tightest cluster for the
# Austin/Boise/Tampa case studies.
AFFORDABILITY_THRESHOLD = 5.0

# Quarters of lag applied to every predictor before it may be used, so no
# feature can see the same quarter that defines the label.
FEATURE_LAG_QUARTERS = 4

# ACS 5-year median household income (table B19013) years to pull.
ACS_YEARS = range(2010, 2025)


def census_api_key() -> str | None:
    """Census API key from the environment, or None if unset.

    Get a free key at https://api.census.gov/data/key_signup.html and export it
    as CENSUS_API_KEY. Never hardcode it -- an earlier version of this project
    committed one and it had to be treated as burned.
    """
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    return key or None


def ensure_dirs() -> None:
    """Create the cache and build directories if they do not exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
