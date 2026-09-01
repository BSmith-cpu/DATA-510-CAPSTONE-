"""Bureau of Labor Statistics sources: local unemployment (LAUS) and wages (QCEW).

These two were the pipeline's reproducibility hole. Both were previously
produced by throwaway parsing that never made it into the repository -- only the
resulting CSVs were committed, so nobody could refresh or audit them. The
extraction logic now lives here.

Both feeds require a real User-Agent header; BLS returns 403 without one, which
is what originally made them look unreachable.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

from ..cache import cache_path, fetch
from ..cbsa import laus_area_to_cbsa, qcew_area_to_cbsa
from ..config import RAW_DIR
from .base import Source

log = logging.getLogger(__name__)

LAUS_BASE = "https://download.bls.gov/pub/time.series/la"
QCEW_BASE = "https://data.bls.gov/cew/data/files"


class Unemployment(Source):
    """Metro unemployment rate from BLS Local Area Unemployment Statistics.

    LAUS ships as fixed-width-ish tab-separated time series files: one mapping
    series IDs to areas, another holding monthly values. Metro rates live under
    area types ``B`` (metropolitan areas) and ``C`` (metropolitan divisions);
    both are kept, matching how the population source handles divisions.
    """

    name = "unemployment"
    description = "BLS LAUS metro unemployment rate (monthly, averaged to quarters)"
    value_columns = ["unemployment_rate"]

    MEASURE_UNEMPLOYMENT_RATE = 3

    def fetch(self, *, refresh: bool = False) -> Path:
        self._series_path = fetch(
            f"{LAUS_BASE}/la.series", "laus_series.tsv", refresh=refresh
        )
        self._data_path = fetch(
            f"{LAUS_BASE}/la.data.60.Metro", "laus_metro.tsv", refresh=refresh
        )
        return self._data_path

    def normalize(self, path: Path) -> pd.DataFrame:
        series = pd.read_csv(self._series_path, sep="\t", dtype=str)
        series.columns = [c.strip() for c in series.columns]
        for column in series.columns:
            series[column] = series[column].astype(str).str.strip()

        series["measure_code"] = pd.to_numeric(series["measure_code"], errors="coerce")
        rates = series[
            series["measure_code"].eq(self.MEASURE_UNEMPLOYMENT_RATE)
            & series["area_type_code"].isin(["B", "C"])
        ].copy()
        rates["cbsa"] = rates["area_code"].map(laus_area_to_cbsa)
        rates = rates.dropna(subset=["cbsa"])

        # Prefer unadjusted series, which have the broader metro coverage; fall
        # back to seasonally adjusted where that is all a metro publishes.
        rates["preference"] = rates["seasonal"].map({"U": 0, "S": 1}).fillna(2)
        rates = rates.sort_values(["cbsa", "preference"]).drop_duplicates(
            subset="cbsa", keep="first"
        )
        crosswalk = rates[["series_id", "cbsa"]].copy()
        crosswalk["cbsa"] = crosswalk["cbsa"].astype(int)

        wanted = set(crosswalk["series_id"])
        chunks = []
        for chunk in pd.read_csv(self._data_path, sep="\t", dtype=str, chunksize=500_000):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk["series_id"] = chunk["series_id"].str.strip()
            hit = chunk[chunk["series_id"].isin(wanted)]
            if not hit.empty:
                chunks.append(hit)

        values = pd.concat(chunks, ignore_index=True)
        # M13 is an annual average, not a month; drop it.
        values = values[values["period"].str.match(r"^M(0[1-9]|1[0-2])$", na=False)].copy()
        values["unemployment_rate"] = pd.to_numeric(
            values["value"].str.strip(), errors="coerce"
        )
        values["year"] = values["year"].astype(int)
        values["qtr"] = ((values["period"].str[1:].astype(int) - 1) // 3) + 1

        merged = values.merge(crosswalk, on="series_id", how="inner")
        return (
            merged.dropna(subset=["unemployment_rate"])
            .groupby(["cbsa", "year", "qtr"], as_index=False)["unemployment_rate"]
            .mean()
        )


class Wages(Source):
    """Metro average weekly wage and employment from BLS QCEW.

    QCEW publishes one ~300MB zipped CSV per year covering every
    county/industry/ownership combination. Only ``agglvl_code == 40`` is needed:
    the metro-level, all-industry, all-ownership aggregate, one row per
    metro-quarter. Downloads are cached, so the cost is paid once per year of
    history rather than once per run.
    """

    name = "wages"
    description = "BLS QCEW metro average weekly wage and employment (quarterly)"
    value_columns = ["qcew_avg_wkly_wage", "qcew_employment"]

    METRO_AGGREGATION_LEVEL = "40"
    YEARS = range(2016, 2026)

    def __init__(self, known_cbsas: set[int] | None = None):
        self._known_cbsas = known_cbsas or set()

    def fetch(self, *, refresh: bool = False) -> Path:
        self._paths = []
        for year in self.YEARS:
            url = f"{QCEW_BASE}/{year}/csv/{year}_qtrly_singlefile.zip"
            try:
                self._paths.append(
                    fetch(url, f"qcew_{year}.zip", refresh=refresh, timeout=900)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("QCEW %s unavailable (%s); skipping", year, exc)
        if not self._paths:
            fallback = RAW_DIR / "qcew_wages_metro.csv"
            if fallback.exists():
                log.warning("no QCEW archives retrieved; using committed extract")
                self._paths = [fallback]
            else:
                raise RuntimeError("no QCEW data available")
        return self._paths[0]

    def normalize(self, path: Path) -> pd.DataFrame:
        # Committed-extract fallback is already in normalized form.
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            return frame[["cbsa", "year", "qtr", *self.value_columns]]

        frames = [self._read_archive(p) for p in self._paths]
        stacked = pd.concat(frames, ignore_index=True)

        known = self._known_cbsas or set(stacked["area_fips"].map(_prefix_guess).dropna())
        stacked["cbsa"] = stacked["area_fips"].map(
            lambda code: qcew_area_to_cbsa(code, known)
        )
        stacked = stacked.dropna(subset=["cbsa"])
        stacked["cbsa"] = stacked["cbsa"].astype(int)

        stacked["qcew_avg_wkly_wage"] = pd.to_numeric(
            stacked["avg_wkly_wage"], errors="coerce"
        )
        stacked["qcew_employment"] = pd.to_numeric(
            stacked["month3_emplvl"], errors="coerce"
        )
        stacked = stacked[stacked["qcew_avg_wkly_wage"] > 0]

        return (
            stacked.groupby(["cbsa", "year", "qtr"], as_index=False)[self.value_columns]
            .mean()
        )

    def _read_archive(self, path: Path) -> pd.DataFrame:
        archive = zipfile.ZipFile(path)
        member = archive.namelist()[0]
        keep = []
        with archive.open(member) as handle:
            for chunk in pd.read_csv(handle, dtype=str, chunksize=500_000):
                metro = chunk[
                    chunk["agglvl_code"].eq(self.METRO_AGGREGATION_LEVEL)
                    & chunk["area_fips"].str.startswith("C", na=False)
                ]
                if not metro.empty:
                    keep.append(
                        metro[
                            ["area_fips", "year", "qtr", "avg_wkly_wage", "month3_emplvl"]
                        ]
                    )
        frame = pd.concat(keep, ignore_index=True)
        frame["year"] = frame["year"].astype(int)
        frame["qtr"] = frame["qtr"].astype(int)
        return frame


def _prefix_guess(area_fips: str) -> int | None:
    """Best-effort CBSA guess used only when no reference universe is supplied."""
    if isinstance(area_fips, str) and area_fips.startswith("C") and area_fips[1:].isdigit():
        return int(area_fips[1:] + "0")
    return None
