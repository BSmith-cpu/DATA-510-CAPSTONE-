"""Tests for leakage safety and label construction.

The leakage test is the important one: an earlier version of this project
reported 0.98 accuracy that turned out to be an artifact, so the property that
features never see the label's own quarter is asserted directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_pipeline.config import AFFORDABILITY_THRESHOLD
from housing_pipeline.features import (
    FEATURE_NAMES,
    FEATURES,
    MONOTONE_CONSTRAINTS,
    build_features,
)
from housing_pipeline.panel import add_affordability_labels


def make_panel(n_quarters: int = 40, n_metros: int = 3) -> pd.DataFrame:
    """A synthetic panel with steadily rising prices and incomes."""
    rows = []
    for metro in range(n_metros):
        cbsa = 10000 + metro
        for i in range(n_quarters):
            year = 2010 + i // 4
            qtr = i % 4 + 1
            rows.append(
                {
                    "cbsa": cbsa,
                    "year": year,
                    "qtr": qtr,
                    "zhvi_qtr": 200_000 * (1.02**i) * (1 + 0.1 * metro),
                    "zori_qtr": 1200 * (1.01**i),
                    "inventory_qtr": 5000 * (1 - 0.005 * i),
                    "index_sa": 100 * (1.015**i),
                    "population": 500_000 * (1.004**i),
                    "median_income": 60_000 * (1.005**i),
                    "unemployment_rate": 5.0 + np.sin(i / 4),
                    "qcew_avg_wkly_wage": 1000 * (1.008**i),
                    "qcew_employment": 250_000 * (1.003**i),
                    "market_qtr": 2000 * (1.02**i),
                }
            )
    return pd.DataFrame(rows)


class TestLabels:
    def test_price_to_income_and_unaffordable_flag(self):
        panel = pd.DataFrame(
            {
                "cbsa": [1, 1],
                "year": [2020, 2020],
                "qtr": [1, 2],
                "zhvi_qtr": [400_000, 200_000],
                "median_income": [50_000, 50_000],
            }
        )
        labelled = add_affordability_labels(panel)
        assert labelled["price_to_income_ratio"].tolist() == [8.0, 4.0]
        assert labelled["is_unaffordable"].tolist() == [True, False]

    def test_onset_fires_only_on_the_transition_quarter(self):
        # ratio: 4, 4, 6, 6 -> onset only at index 2
        panel = pd.DataFrame(
            {
                "cbsa": [1] * 4,
                "year": [2020] * 4,
                "qtr": [1, 2, 3, 4],
                "zhvi_qtr": [200_000, 200_000, 300_000, 300_000],
                "median_income": [50_000] * 4,
            }
        )
        labelled = add_affordability_labels(panel)
        assert labelled["collapse_onset"].tolist() == [False, False, True, False]

    def test_confirmed_onset_requires_the_next_quarter_to_hold(self):
        # A single-quarter spike that reverts is not a confirmed collapse.
        panel = pd.DataFrame(
            {
                "cbsa": [1] * 4,
                "year": [2020] * 4,
                "qtr": [1, 2, 3, 4],
                "zhvi_qtr": [200_000, 300_000, 200_000, 200_000],
                "median_income": [50_000] * 4,
            }
        )
        labelled = add_affordability_labels(panel)
        assert labelled["collapse_onset"].tolist() == [False, True, False, False]
        assert not labelled["collapse_onset_confirmed"].any()

    def test_onset_at_the_edge_of_the_window_is_not_confirmed(self):
        # This is what produced the Springfield MA / Traverse City MI failures:
        # a crossing in the final observed quarter can never be verified.
        panel = pd.DataFrame(
            {
                "cbsa": [1] * 3,
                "year": [2020] * 3,
                "qtr": [1, 2, 3],
                "zhvi_qtr": [200_000, 200_000, 300_000],
                "median_income": [50_000] * 3,
            }
        )
        labelled = add_affordability_labels(panel)
        assert labelled["collapse_onset"].iloc[-1]
        assert not labelled["collapse_onset_confirmed"].iloc[-1]

    def test_metro_with_no_income_data_produces_no_spurious_onsets(self):
        panel = pd.DataFrame(
            {
                "cbsa": [1] * 3,
                "year": [2020] * 3,
                "qtr": [1, 2, 3],
                "zhvi_qtr": [200_000] * 3,
                "median_income": [np.nan] * 3,
            }
        )
        labelled = add_affordability_labels(panel)
        assert not labelled["collapse_onset"].any()
        assert not labelled["collapse_onset_confirmed"].any()


class TestFeatureLeakage:
    def test_every_feature_is_lagged_at_least_four_quarters(self):
        """Perturbing recent quarters must not change earlier feature values.

        If a feature could see its own quarter, editing quarter T would move the
        feature value at quarter T -- which is exactly the leak this asserts
        against.
        """
        panel = add_affordability_labels(make_panel(n_quarters=40, n_metros=1))
        base = build_features(panel)

        tampered = panel.copy()
        # Blow up the last 4 quarters of every input series.
        recent = tampered.index[-4:]
        for column in ("zhvi_qtr", "zori_qtr", "index_sa", "population",
                       "median_income", "unemployment_rate", "inventory_qtr",
                       "qcew_avg_wkly_wage", "qcew_employment", "market_qtr"):
            tampered.loc[recent, column] *= 10
        after = build_features(tampered)

        # Feature rows before the tampered window must be untouched.
        safe = base.index[:-8]
        for feature in FEATURE_NAMES:
            pd.testing.assert_series_equal(
                base.loc[safe, feature],
                after.loc[safe, feature],
                check_names=False,
                obj=f"{feature} changed in rows that predate the perturbation",
            )

    def test_features_do_not_reference_the_current_quarter(self):
        panel = add_affordability_labels(make_panel(n_quarters=30, n_metros=1))
        frame = build_features(panel)
        # price_to_income_lag at row i must equal price_to_income_ratio at i-4.
        ratio = frame["price_to_income_ratio"].to_numpy()
        lagged = frame["price_to_income_lag"].to_numpy()
        np.testing.assert_allclose(lagged[4:], ratio[:-4], rtol=1e-9)


class TestFeatureContract:
    def test_all_declared_features_are_produced(self):
        panel = add_affordability_labels(make_panel())
        frame = build_features(panel)
        missing = [f for f in FEATURE_NAMES if f not in frame.columns]
        assert not missing, f"declared but not produced: {missing}"

    def test_monotone_constraints_align_with_feature_order(self):
        assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_NAMES)
        for feature, constraint in zip(FEATURES, MONOTONE_CONSTRAINTS):
            assert constraint == feature.monotonic

    def test_ambiguous_direction_features_are_unconstrained(self):
        by_name = {f.name: f.monotonic for f in FEATURES}
        for name in ("unemployment_rate_lag", "inv_qoq_lag", "market_yoy_lag",
                     "qcew_wage_yoy_lag", "qcew_emp_yoy_lag"):
            assert by_name[name] == 0, f"{name} should not have a sign imposed"

    def test_features_are_computed_per_metro_not_across_them(self):
        """A metro's features must not bleed in from the previous metro's rows."""
        panel = add_affordability_labels(make_panel(n_quarters=12, n_metros=2))
        frame = build_features(panel)
        # The first 4 rows of the second metro have no history of their own.
        second = frame[frame["cbsa"] == frame["cbsa"].max()].head(4)
        assert second["price_to_income_lag"].isna().all()
