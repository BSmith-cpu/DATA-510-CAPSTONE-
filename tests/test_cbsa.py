"""Tests for metro identity translation.

Every case here corresponds to a bug that actually shipped in this project.
"""

from __future__ import annotations

import pandas as pd
import pytest

from housing_pipeline.cbsa import (
    apply_income_aliases,
    build_name_crosswalk,
    laus_area_to_cbsa,
    match_metro_names,
    month_to_quarter,
    qcew_area_to_cbsa,
    zillow_name_key,
)


class TestZillowNameKey:
    def test_truncates_at_comma_and_hyphen(self):
        # The original join only cut at the comma, so "Austin-Round Rock-San
        # Marcos, TX" never matched Zillow's "Austin, TX".
        assert zillow_name_key("Austin-Round Rock-San Marcos, TX") == ("Austin", "TX")
        assert zillow_name_key("Austin, TX") == ("Austin", "TX")

    def test_multi_state_metro_takes_first_state(self):
        assert zillow_name_key("New York-Newark-Jersey City, NY-NJ-PA") == ("New York", "NY")

    def test_simple_metro(self):
        assert zillow_name_key("Boise City, ID") == ("Boise City", "ID")

    def test_rejects_input_without_a_state(self):
        assert zillow_name_key("United States") == (None, None)
        assert zillow_name_key(None) == (None, None)


class TestNameMatching:
    @pytest.fixture
    def crosswalk(self):
        reference = pd.DataFrame(
            {
                "cbsa": [12420, 14260, 45300, 35620],
                "metro_name": [
                    "Austin-Round Rock-San Marcos, TX",
                    "Boise City, ID",
                    "Tampa-St. Petersburg-Clearwater, FL",
                    "New York-Newark-Jersey City, NY-NJ-PA",
                ],
            }
        )
        return build_name_crosswalk(reference)

    def test_matches_zillow_short_labels(self, crosswalk):
        labels = pd.Series(["Austin, TX", "Boise City, ID", "New York, NY"])
        matched = match_metro_names(labels, crosswalk)
        by_name = dict(zip(matched["region_name"], matched["cbsa"]))
        assert by_name["Austin, TX"] == 12420
        assert by_name["Boise City, ID"] == 14260
        assert by_name["New York, NY"] == 35620

    def test_unmatched_label_is_kept_with_null_rather_than_dropped(self, crosswalk):
        matched = match_metro_names(pd.Series(["Nowhere, ZZ"]), crosswalk)
        assert len(matched) == 1
        assert pd.isna(matched["cbsa"].iloc[0])

    def test_same_city_name_in_different_states_does_not_collide(self):
        reference = pd.DataFrame(
            {
                "cbsa": [18140, 38900],
                "metro_name": ["Columbus, OH", "Portland-Vancouver-Hillsboro, OR-WA"],
            }
        )
        crosswalk = build_name_crosswalk(reference)
        matched = match_metro_names(pd.Series(["Portland, ME"]), crosswalk)
        # Portland, ME must not match Portland, OR.
        assert pd.isna(matched["cbsa"].iloc[0])

    def test_one_row_per_input_label(self, crosswalk):
        labels = pd.Series(["Austin, TX", "Austin, TX", "Boise City, ID"])
        matched = match_metro_names(labels, crosswalk)
        assert len(matched) == 2


class TestQcewAreaCodes:
    def test_expands_truncated_four_digit_code(self):
        assert qcew_area_to_cbsa("C1242", {12420, 14260}) == 12420

    def test_returns_none_when_ambiguous(self):
        # If two real CBSAs shared a prefix, guessing would be worse than
        # dropping the row.
        assert qcew_area_to_cbsa("C1242", {12420, 12421}) is None

    def test_rejects_non_metro_codes(self):
        assert qcew_area_to_cbsa("48015", {12420}) is None   # a county
        assert qcew_area_to_cbsa("US000", {12420}) is None   # national
        assert qcew_area_to_cbsa(None, {12420}) is None


class TestLausAreaCodes:
    def test_parses_metropolitan_statistical_area(self):
        assert laus_area_to_cbsa("MT4812420000000") == 12420

    def test_parses_metropolitan_division(self):
        # Divisions must work too, or the large multi-division metros vanish.
        assert laus_area_to_cbsa("DV0631084000000") == 31084

    def test_rejects_other_area_types(self):
        assert laus_area_to_cbsa("ST0100000000000") is None
        assert laus_area_to_cbsa(None) is None


class TestIncomeAliases:
    def test_tampa_income_is_copied_onto_the_division_code(self):
        # ACS publishes Tampa under 45300; the panel keys it on 45294.
        income = pd.DataFrame({"cbsa": [45300, 12420], "year": [2020, 2020],
                               "median_income": [60000, 80000]})
        aliased = apply_income_aliases(income)
        assert 45294 in set(aliased["cbsa"])
        # The original code is preserved, not reassigned.
        assert 45300 in set(aliased["cbsa"])
        assert aliased.loc[aliased["cbsa"] == 45294, "median_income"].iloc[0] == 60000

    def test_is_a_noop_when_no_aliased_codes_present(self):
        income = pd.DataFrame({"cbsa": [12420], "year": [2020], "median_income": [80000]})
        assert len(apply_income_aliases(income)) == 1


def test_month_to_quarter():
    assert [month_to_quarter(m) for m in (1, 3, 4, 6, 7, 9, 10, 12)] == [
        1, 1, 2, 2, 3, 3, 4, 4
    ]


class TestAnnualInterpolation:
    """The income sawtooth fix.

    A step-broadcast made price_to_income_ratio drop ~1.9% every Q1 and climb
    Q2-Q4, purely because ACS income is annual. That artifact was large enough
    that confirmed onsets never occurred in Q1.
    """

    @pytest.fixture
    def annual(self):
        return pd.DataFrame(
            {"cbsa": [1, 1, 1], "year": [2020, 2021, 2022],
             "median_income": [50_000, 60_000, 70_000]}
        )

    def test_expands_to_four_quarters_per_year(self, annual):
        from housing_pipeline.cbsa import interpolate_annual_to_quarterly
        out = interpolate_annual_to_quarterly(annual, "median_income")
        assert len(out) == 12
        assert sorted(out["qtr"].unique()) == [1, 2, 3, 4]

    def test_anchor_quarter_keeps_the_reported_annual_value(self, annual):
        from housing_pipeline.cbsa import interpolate_annual_to_quarterly
        out = interpolate_annual_to_quarterly(annual, "median_income", anchor_qtr=3)
        anchored = out[(out["year"] == 2021) & (out["qtr"] == 3)]
        assert anchored["median_income"].iloc[0] == pytest.approx(60_000)

    def test_values_move_smoothly_instead_of_stepping(self, annual):
        from housing_pipeline.cbsa import interpolate_annual_to_quarterly
        out = interpolate_annual_to_quarterly(annual, "median_income")
        out = out.sort_values(["year", "qtr"])
        # Between the 2020 and 2021 anchors every quarter should differ.
        middle = out[(out["year"] == 2021)]["median_income"]
        assert middle.nunique() > 1, "income still constant within a year"

    def test_no_quarter_to_quarter_jump_larger_than_the_annual_change(self, annual):
        from housing_pipeline.cbsa import interpolate_annual_to_quarterly
        out = interpolate_annual_to_quarterly(annual, "median_income")
        out = out.sort_values(["year", "qtr"])
        steps = out["median_income"].diff().dropna().abs()
        # Annual change is 10k over 4 quarters, so no single step should approach it.
        assert steps.max() < 10_000

    def test_ends_are_held_flat_not_extrapolated(self, annual):
        from housing_pipeline.cbsa import interpolate_annual_to_quarterly
        out = interpolate_annual_to_quarterly(annual, "median_income").sort_values(
            ["year", "qtr"]
        )
        # Before the first anchor, income should not fall below the first value.
        assert out["median_income"].iloc[0] == pytest.approx(50_000)
        assert out["median_income"].iloc[-1] == pytest.approx(70_000)

    def test_metros_are_interpolated_independently(self):
        from housing_pipeline.cbsa import interpolate_annual_to_quarterly
        annual = pd.DataFrame(
            {"cbsa": [1, 1, 2, 2], "year": [2020, 2021, 2020, 2021],
             "median_income": [50_000, 60_000, 90_000, 90_000]}
        )
        out = interpolate_annual_to_quarterly(annual, "median_income")
        # Metro 2 is flat, so it must stay flat regardless of metro 1's ramp.
        assert out[out["cbsa"] == 2]["median_income"].nunique() == 1
