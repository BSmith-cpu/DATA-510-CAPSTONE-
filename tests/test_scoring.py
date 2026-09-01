"""Tests for watchlist construction and prior correction.

The central property: correcting for the base-rate shift must lower the scores
without reordering them. Ranking is what cross-validation actually validated, so
it must survive calibration untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_pipeline.scoring import (
    BaseRates,
    assign_tiers,
    base_rates,
    build_watchlist,
    deployment_base_rate,
    prior_correct,
    watchlist_summary,
)


class TestBaseRates:
    def test_measures_the_deployment_rate_from_observed_data(self):
        at_risk = pd.DataFrame({"collapse_onset_confirmed": [True] + [False] * 99})
        assert deployment_base_rate(at_risk) == pytest.approx(0.01)

    def test_enrichment_is_the_ratio_of_the_two_rates(self):
        rates = BaseRates(training=0.04, deployment=0.01)
        assert rates.enrichment == pytest.approx(4.0)

    def test_empty_population_is_an_explicit_error(self):
        with pytest.raises(ValueError, match="empty population"):
            deployment_base_rate(pd.DataFrame({"collapse_onset_confirmed": []}))

    def test_computes_both_rates_together(self):
        pool = pd.DataFrame({"collapse_onset_confirmed": [True] * 4 + [False] * 96})
        at_risk = pd.DataFrame({"collapse_onset_confirmed": [True] + [False] * 99})
        rates = base_rates(pool, at_risk)
        assert rates.training == pytest.approx(0.04)
        assert rates.deployment == pytest.approx(0.01)


class TestPriorCorrection:
    def test_lowers_scores_when_deployment_is_rarer_than_training(self):
        rates = BaseRates(training=0.04, deployment=0.01)
        raw = np.array([0.1, 0.3, 0.6, 0.9])
        corrected = prior_correct(raw, rates)
        assert np.all(corrected < raw)

    def test_preserves_ordering(self):
        """The whole point: calibration must not reshuffle the watchlist."""
        rates = BaseRates(training=0.04, deployment=0.01)
        raw = np.array([0.02, 0.31, 0.15, 0.88, 0.44])
        corrected = prior_correct(raw, rates)
        assert list(np.argsort(raw)) == list(np.argsort(corrected))

    def test_is_a_noop_when_the_priors_match(self):
        rates = BaseRates(training=0.05, deployment=0.05)
        raw = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(prior_correct(raw, rates), raw, atol=1e-6)

    def test_raises_scores_when_deployment_is_denser(self):
        rates = BaseRates(training=0.01, deployment=0.04)
        raw = np.array([0.1, 0.5])
        assert np.all(prior_correct(raw, rates) > raw)

    def test_output_stays_a_valid_probability(self):
        rates = BaseRates(training=0.5, deployment=0.001)
        corrected = prior_correct(np.array([0.0, 1e-12, 0.5, 1 - 1e-12, 1.0]), rates)
        assert np.all((corrected >= 0.0) & (corrected <= 1.0))
        assert not np.isnan(corrected).any()


class TestTiers:
    def test_bands_follow_percentile_not_raw_value(self):
        percentile = pd.Series([0.99, 0.90, 0.50, 0.01])
        assert assign_tiers(percentile).tolist() == [
            "Elevated", "Watch", "Monitor", "Monitor"
        ]


class TestWatchlist:
    @pytest.fixture
    def holdout(self):
        # Three metros, two quarters each.
        return pd.DataFrame(
            {
                "cbsa": [1, 1, 2, 2, 3, 3],
                "metro_name": ["A", "A", "B", "B", "C", "C"],
            }
        )

    def test_one_row_per_metro_ranked_high_to_low(self, holdout):
        scores = [0.1, 0.1, 0.9, 0.9, 0.5, 0.5]
        watchlist = build_watchlist(holdout, scores)
        assert len(watchlist) == 3
        assert watchlist["metro_name"].tolist() == ["B", "C", "A"]
        assert watchlist["risk_rank"].tolist() == [1, 2, 3]

    def test_metro_score_is_the_mean_of_its_quarters(self, holdout):
        scores = [0.2, 0.4, 0.9, 0.9, 0.5, 0.5]
        watchlist = build_watchlist(holdout, scores)
        a = watchlist[watchlist["metro_name"] == "A"]
        assert a["risk_score_raw"].iloc[0] == pytest.approx(0.3)

    def test_probability_only_appears_when_base_rates_are_supplied(self, holdout):
        scores = [0.1, 0.1, 0.9, 0.9, 0.5, 0.5]
        assert "risk_probability" not in build_watchlist(holdout, scores).columns

        rates = BaseRates(training=0.04, deployment=0.01)
        with_prob = build_watchlist(holdout, scores, rates)
        assert "risk_probability" in with_prob.columns

    def test_calibration_does_not_change_the_ranking(self, holdout):
        scores = [0.1, 0.1, 0.9, 0.9, 0.5, 0.5]
        rates = BaseRates(training=0.04, deployment=0.01)
        plain = build_watchlist(holdout, scores)
        calibrated = build_watchlist(holdout, scores, rates)
        assert plain["metro_name"].tolist() == calibrated["metro_name"].tolist()

    def test_summary_flags_how_the_scores_should_be_read(self, holdout):
        rates = BaseRates(training=0.04, deployment=0.01)
        watchlist = build_watchlist(holdout, [0.1, 0.1, 0.9, 0.9, 0.5, 0.5], rates)
        summary = watchlist_summary(watchlist, top_n=3)
        assert "relative standing" in summary
        assert "prior-corrected" in summary
