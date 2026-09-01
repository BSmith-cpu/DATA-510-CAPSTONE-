"""Zillow Research metro products: home values, rents, and for-sale inventory.

All three ship the same shape -- one row per metro, one column per month -- so
they share a single implementation and differ only in URL and output column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cache import fetch
from ..cbsa import build_name_crosswalk, match_metro_names
from ..config import RAW_DIR
from .base import Source, quarterly_mean

ZILLOW_BASE = "https://files.zillowstatic.com/research/public_csvs"


class _ZillowMonthly(Source):
    """Shared loader for Zillow's wide monthly metro CSVs."""

    url: str
    cache_name: str
    fallback_name: str | None = None
    out_column: str

    def __init__(self, reference: pd.DataFrame):
        # The Census population table is the naming authority these free-text
        # metro labels get matched against.
        self._crosswalk = build_name_crosswalk(reference)
        self.matched_metros = 0
        self.total_metros = 0

    def fetch(self, *, refresh: bool = False) -> Path:
        fallback = RAW_DIR / self.fallback_name if self.fallback_name else None
        return fetch(self.url, self.cache_name, refresh=refresh, fallback=fallback)

    def normalize(self, path: Path) -> pd.DataFrame:
        wide = pd.read_csv(path)

        id_cols = [c for c in wide.columns if not _looks_like_date(c)]
        date_cols = [c for c in wide.columns if _looks_like_date(c)]

        long = wide.melt(
            id_vars=id_cols,
            value_vars=date_cols,
            var_name="date",
            value_name="value",
        )
        long["date"] = pd.to_datetime(long["date"], errors="coerce")
        long = long.dropna(subset=["date", "value"])

        names = match_metro_names(long["RegionName"], self._crosswalk)
        self.total_metros = len(names)
        self.matched_metros = int(names["cbsa"].notna().sum())

        long = long.merge(
            names.rename(columns={"region_name": "RegionName"}),
            on="RegionName",
            how="left",
        )
        long = long.dropna(subset=["cbsa"])

        quarterly = quarterly_mean(
            long,
            group_cols=["cbsa"],
            date_col="date",
            value_col="value",
            out_col=self.out_column,
        )
        quarterly["cbsa"] = quarterly["cbsa"].astype("int64")
        # One metro can match several Zillow labels; average rather than fan out.
        return (
            quarterly.groupby(["cbsa", "year", "qtr"], as_index=False)[self.out_column]
            .mean()
        )

    def coverage(self, frame: pd.DataFrame) -> str:
        return (
            f"{frame['cbsa'].nunique()} metros, {len(frame)} rows "
            f"({self.matched_metros}/{self.total_metros} labels matched to a CBSA)"
        )


def _looks_like_date(column: str) -> bool:
    return bool(pd.to_datetime(column, errors="coerce") is not pd.NaT
                and not pd.isna(pd.to_datetime(column, errors="coerce")))


class ZHVI(_ZillowMonthly):
    name = "zhvi"
    description = "Zillow Home Value Index (typical home value), monthly by metro"
    value_columns = ["zhvi_qtr"]
    url = f"{ZILLOW_BASE}/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    cache_name = "zillow_zhvi.csv"
    fallback_name = "zhvi_raw.csv"
    out_column = "zhvi_qtr"


class ZORI(_ZillowMonthly):
    name = "zori"
    description = "Zillow Observed Rent Index, monthly by metro"
    value_columns = ["zori_qtr"]
    url = f"{ZILLOW_BASE}/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv"
    cache_name = "zillow_zori.csv"
    fallback_name = "zori_raw.csv"
    out_column = "zori_qtr"


class Inventory(_ZillowMonthly):
    name = "inventory"
    description = "Zillow for-sale inventory (housing supply), monthly by metro"
    value_columns = ["inventory_qtr"]
    url = f"{ZILLOW_BASE}/invt_fs/Metro_invt_fs_uc_sfrcondo_sm_month.csv"
    cache_name = "zillow_inventory.csv"
    fallback_name = "zillow_inventory_raw.csv"
    out_column = "inventory_qtr"
