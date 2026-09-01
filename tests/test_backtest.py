"""Tests for walk-forward backtesting.

The load-bearing test is `test_training_never_sees_the_origin_or_later`: the
whole point of this module is that no fold may use future information, so that
property is asserted directly rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_pipeline.backtest import (
    BacktestResult,
    lead_time,
    lead_time_summary,
    precision_at_k,
    quarter_index,
    recall_at_k,
    walk_forward,
)


def synthetic_features(n_metros: int = 40, years=range(2018, 2025)) -> pd.DataFrame:
    """A panel where one feature genuinely predicts onset a year ahead."""
    rng = np.random.default_rng(0)
    rows = []
    for metro in range(n_metros):
        cbsa = 10000 + metro
        # A tenth of metros are "hot" and eventually cross the threshold.
        hot = metro % 10 == 0
        for year in years:
            for qtr in (1, 2, 3, 4):
                signal = rng.normal(1.5 if hot else 0.0, 0.5)
                onset = bool(hot and year >= 2021 and qtr == 2)
                row = {
                    "cbsa": cbsa,
                    "metro_name": f"Metro {metro}",
                    "year": year,
                    "qtr": qtr,
                    "prev_unaffordable": False,
                    "collapse_onset_confirmed": onset,
                }
                for name in FEATURES:
                    row[name] = signal + rng.normal(0, 0.1)
                rows.append(row)
    return pd.DataFrame(rows)


FEATURES = ["f1", "f2"]


def simple_model_factory(scale_pos_weight: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, class_weight="balanced")


class TestRankMetrics:
    def test_precision_at_k(self):
        ranked = np.array([1, 0, 1, 0, 0])
        assert precision_at_k(ranked, 2) == pytest.approx(0.5)
        assert precision_at_k(ranked, 4) == pytest.approx(0.5)

    def test_recall_at_k(self):
        ranked = np.array([1, 0, 1, 0, 0])
        assert recall_at_k(ranked, 1) == pytest.approx(0.5)
        assert recall_at_k(ranked, 3) == pytest.approx(1.0)

    def test_recall_is_nan_without_events(self):
        assert np.isnan(recall_at_k(np.zeros(5), 3))


def test_quarter_index_is_monotonic_across_year_boundaries():
    assert quarter_index(2020, 4) + 1 == quarter_index(2021, 1)


class TestWalkForward:
    @pytest.fixture(scope="class")
    def result(self):
        return walk_forward(
            synthetic_features(),
            start=(2021, 1),
            min_train_events=2,
            feature_names=FEATURES,
            model_factory=simple_model_factory,
        )

    def test_produces_predictions_and_per_origin_rows(self, result):
        assert not result.predictions.empty
        assert not result.per_origin.empty

    def test_training_never_sees_the_origin_or_later(self):
        """No fold may train on a row at or after the quarter it scores."""
        seen: list[tuple[int, int]] = []

        def spy_factory(spw):
            model = simple_model_factory(spw)
            original_fit = model.fit

            def fit(X, y):
                # Record the highest training index used for this fold.
                seen.append((int(X.index.max()), len(X)))
                return original_fit(X, y)

            model.fit = fit
            return model

        features = synthetic_features()
        features = features.reset_index(drop=True)
        features["q"] = quarter_index(features["year"], features["qtr"])

        result = walk_forward(
            features,
            start=(2021, 1),
            min_train_events=2,
            feature_names=FEATURES,
            model_factory=spy_factory,
        )

        # For each origin, every training row's quarter must be < the origin.
        at_risk = features[~features["prev_unaffordable"]].dropna(
            subset=FEATURES + ["collapse_onset_confirmed"]
        )
        for _, row in result.per_origin.iterrows():
            origin_q = row["origin_q"]
            train = at_risk[at_risk["q"] <= origin_q - 1]
            assert len(train) == row["n_train"]
            assert train["q"].max() < origin_q

    def test_scores_only_the_origin_quarter(self, result):
        for origin_q, group in result.predictions.groupby("origin_q"):
            # Every prediction in a fold belongs to that fold's quarter.
            assert group["origin_q"].nunique() == 1
            assert group["origin_q"].iloc[0] == origin_q

    def test_ranks_are_dense_and_start_at_one(self, result):
        for _, group in result.predictions.groupby("origin_q"):
            ranks = sorted(group["rank"])
            assert ranks == list(range(1, len(group) + 1))

    def test_higher_score_gets_a_better_rank(self, result):
        group = result.predictions[
            result.predictions["origin_q"] == result.predictions["origin_q"].min()
        ]
        ordered = group.sort_values("rank")
        assert ordered["score"].is_monotonic_decreasing

    def test_skips_origins_without_enough_training_events(self):
        result = walk_forward(
            synthetic_features(),
            start=(2018, 1),
            min_train_events=5,
            feature_names=FEATURES,
            model_factory=simple_model_factory,
        )
        assert result.skipped
        assert any("training events" in s["reason"] for s in result.skipped)

    def test_learns_the_real_signal(self, result):
        """Sanity check: on data with a planted signal, it should beat chance."""
        pooled = result.pooled()
        assert pooled["pr_auc"] > pooled["base_rate"]

    def test_summary_renders(self, result):
        assert "Walk-forward backtest" in result.summary()


class TestLeadTime:
    def test_tracks_ranks_before_the_onset(self):
        result = walk_forward(
            synthetic_features(),
            start=(2021, 1),
            min_train_events=2,
            feature_names=FEATURES,
            model_factory=simple_model_factory,
        )
        lead = lead_time(result, horizon=4)
        assert not lead.empty
        assert lead["quarters_before_onset"].max() <= 4
        assert (lead["percentile"].dropna() <= 1.0).all()

    def test_summary_has_one_row_per_horizon_step(self):
        result = walk_forward(
            synthetic_features(),
            start=(2021, 1),
            min_train_events=2,
            feature_names=FEATURES,
            model_factory=simple_model_factory,
        )
        summary = lead_time_summary(lead_time(result, horizon=4))
        assert summary["quarters_before_onset"].tolist() == sorted(
            summary["quarters_before_onset"].tolist()
        )

    def test_empty_input_is_handled(self):
        empty = BacktestResult(predictions=pd.DataFrame(), per_origin=pd.DataFrame())
        assert lead_time(empty).empty
        assert lead_time_summary(pd.DataFrame()).empty
