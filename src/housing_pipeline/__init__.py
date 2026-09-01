"""Metro housing affordability pipeline.

Typical use from a notebook:

    from housing_pipeline import load_panel, build_features, FEATURE_NAMES

    panel = load_panel()
    frame = build_features(panel)

Or from the command line:

    python -m housing_pipeline build
"""

from .config import AFFORDABILITY_THRESHOLD, FEATURE_LAG_QUARTERS
from .features import (
    FEATURE_NAMES,
    FEATURES,
    MONOTONE_CONSTRAINTS,
    build_features,
    feature_coverage,
)
from .panel import add_affordability_labels, build_panel, load_panel, save_panel

__version__ = "1.0.0"

__all__ = [
    "AFFORDABILITY_THRESHOLD",
    "FEATURE_LAG_QUARTERS",
    "FEATURES",
    "FEATURE_NAMES",
    "MONOTONE_CONSTRAINTS",
    "add_affordability_labels",
    "build_features",
    "build_panel",
    "feature_coverage",
    "load_panel",
    "save_panel",
]
