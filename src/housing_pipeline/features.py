"""Leakage-safe feature engineering.

Every predictor is lagged four quarters before use, so no feature can observe
the same quarter that defines the label. The lag is applied *after* any
percent-change or rolling calculation, which is the ordering that matters: a
year-over-year change computed on already-lagged values would reach a full year
further back than intended.

Feature direction is declared alongside each definition. Features with an
unambiguous relationship to risk get a monotonic constraint in the model;
features whose direction is genuinely arguable are left unconstrained rather
than having a sign imposed on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FEATURE_LAG_QUARTERS


@dataclass(frozen=True)
class Feature:
    """One model input and what we are willing to assume about it."""

    name: str
    description: str
    # +1 when higher values should mean higher risk, 0 when the direction is
    # not obvious enough to constrain.
    monotonic: int = 1
    source: str = ""


FEATURES: list[Feature] = [
    Feature("price_to_income_lag", "Price-to-income ratio, lagged", 1, "zhvi+acs"),
    Feature("price_to_income_5yr_chg", "5-year change in price-to-income", 1, "zhvi+acs"),
    Feature("zhvi_yoy_lag", "Home value growth, year over year", 1, "zhvi"),
    Feature("zhvi_qoq_lag", "Home value growth, quarter over quarter", 1, "zhvi"),
    Feature(
        "three-year_home_price_growth_trend",
        "Slope of 3-year home-value growth trend",
        1,
        "zhvi",
    ),
    Feature("hpi_yoy_lag", "House price index growth, year over year", 1, "hpi"),
    Feature("hpi_3yr_chg_lag", "House price index 3-year change", 1, "hpi"),
    Feature("pop_velocity_lag", "Change in population growth rate", 1, "population"),
    Feature("pop_acceleration_lag", "Change in population velocity", 1, "population"),
    Feature("zori_yoy_lag", "Rent growth, year over year", 1, "zori"),
    # Direction is genuinely arguable: a hot labor market can outrun incomes,
    # while a downturn drags prices and incomes down together.
    Feature("unemployment_rate_lag", "Local unemployment rate", 0, "unemployment"),
    # More supply should ease prices, but rising inventory can equally signal a
    # market that has already priced people out.
    Feature("inv_qoq_lag", "For-sale inventory, quarter over quarter", 0, "inventory"),
    Feature("sp500_yoy_lag", "S&P 500 growth, year over year", 0, "sp500"),
    Feature("qcew_wage_yoy_lag", "Local wage growth, year over year", 0, "wages"),
    Feature("qcew_emp_yoy_lag", "Local employment growth, year over year", 0, "wages"),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
MONOTONE_CONSTRAINTS: tuple[int, ...] = tuple(f.monotonic for f in FEATURES)


def _lagged_change(
    frame: pd.DataFrame, column: str, periods: int, lag: int
) -> pd.Series:
    """Percent change over `periods`, then shifted back by `lag` quarters."""
    change = frame.groupby("cbsa")[column].transform(lambda s: s.pct_change(periods))
    return change.groupby(frame["cbsa"]).shift(lag)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over a rolling window, NaN unless the window is full."""
    return series.rolling(window, min_periods=window).apply(
        lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if y.notna().all() else np.nan,
        raw=False,
    )


def build_features(panel: pd.DataFrame, lag: int = FEATURE_LAG_QUARTERS) -> pd.DataFrame:
    """Add every model feature to the panel."""
    frame = panel.sort_values(["cbsa", "year", "qtr"]).reset_index(drop=True)
    by_cbsa = frame.groupby("cbsa")

    # Affordability level and its medium-run trajectory.
    frame["price_to_income_lag"] = by_cbsa["price_to_income_ratio"].shift(lag)
    frame["price_to_income_5yr_chg"] = frame.groupby("cbsa")[
        "price_to_income_lag"
    ].transform(lambda s: s - s.shift(20))

    # Home value momentum.
    frame["zhvi_yoy_lag"] = _lagged_change(frame, "zhvi_qtr", 4, lag)
    frame["zhvi_qoq_lag"] = _lagged_change(frame, "zhvi_qtr", 1, lag)
    frame["three-year_home_price_growth_trend"] = frame.groupby("cbsa")[
        "zhvi_yoy_lag"
    ].transform(lambda s: _rolling_slope(s, 12))

    # House price index momentum.
    frame["hpi_yoy_lag"] = _lagged_change(frame, "index_sa", 4, lag)
    hpi_3yr = by_cbsa["index_sa"].transform(lambda s: (s / s.shift(12) - 1) * 100)
    frame["hpi_3yr_chg_lag"] = hpi_3yr.groupby(frame["cbsa"]).shift(lag)

    # Population dynamics: growth rate, then its first and second differences.
    pop_yoy = by_cbsa["population"].transform(lambda s: s.pct_change(4))
    pop_velocity = pop_yoy.groupby(frame["cbsa"]).diff()
    frame["pop_velocity_lag"] = pop_velocity.groupby(frame["cbsa"]).shift(lag)
    frame["pop_acceleration_lag"] = (
        pop_velocity.groupby(frame["cbsa"]).diff().groupby(frame["cbsa"]).shift(lag)
    )

    # Rent, labor market, supply, and macro.
    frame["zori_yoy_lag"] = _lagged_change(frame, "zori_qtr", 4, lag)
    frame["unemployment_rate_lag"] = by_cbsa["unemployment_rate"].shift(lag)
    # Inventory uses quarter-over-quarter rather than year-over-year: the series
    # only starts in 2018, and a YoY-then-lag combination needs eight quarters of
    # history, which would leave the earliest onset events undefined.
    frame["inv_qoq_lag"] = _lagged_change(frame, "inventory_qtr", 1, lag)
    frame["sp500_yoy_lag"] = _lagged_change(frame, "sp500_qtr", 4, lag)
    frame["qcew_wage_yoy_lag"] = _lagged_change(frame, "qcew_avg_wkly_wage", 4, lag)
    frame["qcew_emp_yoy_lag"] = _lagged_change(frame, "qcew_employment", 4, lag)

    return frame


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-feature metro and row coverage -- what is actually usable for modeling."""
    rows = []
    for feature in FEATURES:
        if feature.name not in frame.columns:
            rows.append({"feature": feature.name, "metros": 0, "rows": 0})
            continue
        present = frame[feature.name].notna()
        rows.append(
            {
                "feature": feature.name,
                "metros": int(frame.loc[present, "cbsa"].nunique()),
                "rows": int(present.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("metros", ascending=False)
