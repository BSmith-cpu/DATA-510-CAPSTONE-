"""Tests for the source contract and the panel join.

These run offline against synthetic frames -- they check the wiring, not the
upstream data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from housing_pipeline.panel import _join
from housing_pipeline.sources import SOURCE_NAMES, build_sources
from housing_pipeline.sources.base import Source


class TestSourceRegistry:
    def test_every_registered_name_builds(self):
        sources = build_sources()
        assert set(sources) == set(SOURCE_NAMES)

    def test_every_source_declares_its_contract(self):
        for name, source in build_sources().items():
            assert source.name == name
            assert source.description, f"{name} has no description"
            assert source.value_columns, f"{name} declares no value columns"

    def test_only_the_market_index_is_national(self):
        national = [n for n, s in build_sources().items() if s.national]
        assert national == ["market"]


class _FakeSource(Source):
    """Minimal source used to exercise validation without touching the network."""

    def __init__(self, name, frame, value_columns, national=False):
        self.name = name
        self.description = "fake"
        self.value_columns = value_columns
        self.national = national
        self._frame = frame

    def fetch(self, *, refresh: bool = False):
        return None

    def normalize(self, path):
        return self._frame


class TestSourceValidation:
    def test_rejects_missing_key_columns(self):
        source = _FakeSource("bad", pd.DataFrame({"cbsa": [1], "year": [2020]}), ["v"])
        with pytest.raises(ValueError, match="missing key column"):
            source.load()

    def test_rejects_undelivered_value_columns(self):
        frame = pd.DataFrame({"cbsa": [1], "year": [2020], "qtr": [1]})
        source = _FakeSource("bad", frame, ["missing_col"])
        with pytest.raises(ValueError, match="did not\\s+produce|did not produce"):
            source.load()

    def test_rejects_duplicate_keys_that_would_fan_out_the_join(self):
        frame = pd.DataFrame(
            {"cbsa": [1, 1], "year": [2020, 2020], "qtr": [1, 1], "v": [1.0, 2.0]}
        )
        source = _FakeSource("bad", frame, ["v"])
        with pytest.raises(ValueError, match="duplicate rows"):
            source.load()

    def test_accepts_a_well_formed_frame(self):
        frame = pd.DataFrame(
            {"cbsa": [1, 1], "year": [2020, 2020], "qtr": [1, 2], "v": [1.0, 2.0]}
        )
        assert len(_FakeSource("good", frame, ["v"]).load()) == 2


class TestPanelJoin:
    def test_national_source_broadcasts_to_every_metro(self):
        hpi = pd.DataFrame(
            {
                "cbsa": [1, 2],
                "year": [2020, 2020],
                "qtr": [1, 1],
                "index_sa": [100.0, 110.0],
                "metro_name": ["A", "B"],
            }
        )
        market = pd.DataFrame({"year": [2020], "qtr": [1], "market_qtr": [3000.0]})
        sources = {
            "hpi": _FakeSource("hpi", hpi, ["index_sa", "metro_name"]),
            "market": _FakeSource("market", market, ["market_qtr"], national=True),
        }
        joined = _join({"hpi": hpi, "market": market}, sources)
        assert len(joined) == 2
        assert joined["market_qtr"].tolist() == [3000.0, 3000.0]

    def test_join_raises_rather_than_silently_duplicating_rows(self):
        hpi = pd.DataFrame(
            {"cbsa": [1], "year": [2020], "qtr": [1], "index_sa": [100.0],
             "metro_name": ["A"]}
        )
        # Two rows for the same key would fan the panel out.
        bad = pd.DataFrame(
            {"cbsa": [1, 1], "year": [2020, 2020], "qtr": [1, 1], "v": [1.0, 2.0]}
        )
        sources = {
            "hpi": _FakeSource("hpi", hpi, ["index_sa", "metro_name"]),
            "bad": _FakeSource("bad", bad, ["v"]),
        }
        with pytest.raises(RuntimeError, match="changed the row count"):
            _join({"hpi": hpi, "bad": bad}, sources)

    def test_missing_spine_is_an_explicit_error(self):
        with pytest.raises(RuntimeError, match="without the HPI spine"):
            _join({}, {})
