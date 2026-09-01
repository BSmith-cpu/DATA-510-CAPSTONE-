"""Tests for staleness detection and build-over-build comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from housing_pipeline.freshness import (
    CADENCES,
    check_freshness,
    compare_builds,
    compare_watchlists,
)


def panel_with(latest: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Build a panel where each named column stops at a given year/quarter."""
    rows = []
    for year in range(2020, 2026):
        for qtr in (1, 2, 3, 4):
            row = {"cbsa": 1, "year": year, "qtr": qtr}
            for column, (last_year, last_qtr) in latest.items():
                stop = last_year * 4 + last_qtr
                row[column] = 1.0 if year * 4 + qtr <= stop else None
            rows.append(row)
    return pd.DataFrame(rows)


class TestFreshness:
    def test_current_source_is_not_stale(self):
        panel = panel_with({"zhvi_qtr": (2025, 4)})
        report = check_freshness(panel, as_of=(2025, 4))
        row = report.to_frame().set_index("source").loc["zhvi"]
        assert row["lag_quarters"] == 0
        assert not row["stale"]

    def test_source_far_behind_its_cadence_is_stale(self):
        panel = panel_with({"zhvi_qtr": (2022, 1)})
        report = check_freshness(panel, as_of=(2025, 4))
        assert "zhvi" in report.stale

    def test_acs_lagging_two_years_is_normal_not_stale(self):
        """ACS runs 1-2 years behind by design; that must not raise an alarm."""
        panel = panel_with({"median_income": (2024, 1)})
        report = check_freshness(panel, as_of=(2025, 4))
        row = report.to_frame().set_index("source").loc["acs_income"]
        assert row["lag_quarters"] == 7
        assert not row["stale"], "ACS at its normal lag should not be flagged"

    def test_same_lag_is_stale_for_a_fast_moving_source(self):
        """Seven quarters behind is fine for ACS and broken for Zillow."""
        panel = panel_with({"zhvi_qtr": (2024, 1)})
        report = check_freshness(panel, as_of=(2025, 4))
        assert "zhvi" in report.stale

    def test_missing_column_is_reported_as_stale_not_skipped(self):
        report = check_freshness(panel_with({"zhvi_qtr": (2025, 4)}), as_of=(2025, 4))
        assert "wages" in report.stale

    def test_every_source_has_a_declared_cadence(self):
        from housing_pipeline.sources import SOURCE_NAMES
        assert set(CADENCES) == set(SOURCE_NAMES)

    def test_report_renders(self):
        report = check_freshness(panel_with({"zhvi_qtr": (2025, 4)}), as_of=(2025, 4))
        assert "source" in str(report)


class TestCompareBuilds:
    def test_detects_metros_dropping_out(self):
        previous = pd.DataFrame({"cbsa": [1, 2, 3], "zhvi_qtr": [1.0, 1.0, 1.0]})
        current = pd.DataFrame({"cbsa": [1, 2], "zhvi_qtr": [1.0, 1.0]})
        diff = compare_builds(previous, current).set_index("metric")
        assert diff.loc["metros", "change"] == -1

    def test_detects_coverage_collapse_in_one_source(self):
        previous = pd.DataFrame({"cbsa": [1, 2], "zhvi_qtr": [1.0, 1.0]})
        current = pd.DataFrame({"cbsa": [1, 2], "zhvi_qtr": [1.0, None]})
        diff = compare_builds(previous, current).set_index("metric")
        assert diff.loc["metros_with_zhvi_qtr", "change"] == -1

    def test_tracks_confirmed_onset_count(self):
        previous = pd.DataFrame({"cbsa": [1, 2], "collapse_onset_confirmed": [True, True]})
        current = pd.DataFrame({"cbsa": [1, 2], "collapse_onset_confirmed": [True, False]})
        diff = compare_builds(previous, current).set_index("metric")
        assert diff.loc["confirmed_onsets", "change"] == -1


class TestCompareWatchlists:
    def test_surfaces_large_rank_movement(self):
        previous = pd.DataFrame({"cbsa": [1, 2], "risk_rank": [1, 40]})
        current = pd.DataFrame(
            {"cbsa": [1, 2], "metro_name": ["A", "B"], "risk_rank": [2, 1]}
        )
        moved = compare_watchlists(previous, current, top_n=25)
        b = moved[moved["cbsa"] == 2].iloc[0]
        assert b["rank_change"] == 39  # climbed 39 places

    def test_new_metro_has_no_previous_rank(self):
        previous = pd.DataFrame({"cbsa": [1], "risk_rank": [1]})
        current = pd.DataFrame(
            {"cbsa": [1, 2], "metro_name": ["A", "B"], "risk_rank": [1, 2]}
        )
        moved = compare_watchlists(previous, current)
        assert moved[moved["cbsa"] == 2]["previous_rank"].isna().all()
