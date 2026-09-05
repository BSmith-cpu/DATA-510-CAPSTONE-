"""Metro housing affordability pipeline.

Typical use from a notebook:

    from housing_pipeline import load_panel, build_features, FEATURE_NAMES

    panel = load_panel()
    frame = build_features(panel)

Or from the command line:

    python -m housing_pipeline build
"""

from .backtest import lead_time, lead_time_summary, walk_forward
from .config import AFFORDABILITY_THRESHOLD, FEATURE_LAG_QUARTERS
from .features import (
    FEATURE_NAMES,
    constraints_for,
    FEATURES,
    MONOTONE_CONSTRAINTS,
    build_features,
    feature_coverage,
)
from .panel import add_affordability_labels, build_panel, load_panel, save_panel
from .scoring import (
    BaseRates,
    base_rates,
    build_watchlist,
    deployment_base_rate,
    prior_correct,
    watchlist_summary,
)

__version__ = "1.0.0"

__all__ = [
    "AFFORDABILITY_THRESHOLD",
    "FEATURE_LAG_QUARTERS",
    "FEATURES",
    "FEATURE_NAMES",
    "constraints_for",
    "MONOTONE_CONSTRAINTS",
    "BaseRates",
    "lead_time",
    "lead_time_summary",
    "walk_forward",
    "add_affordability_labels",
    "base_rates",
    "build_features",
    "build_panel",
    "build_watchlist",
    "deployment_base_rate",
    "feature_coverage",
    "prior_correct",
    "watchlist_summary",
    "load_panel",
    "save_panel",
]
