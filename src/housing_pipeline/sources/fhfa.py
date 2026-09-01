"""FHFA House Price Index at the metro level.

Two series are published: all-transactions (broader history) and purchase-only
(cleaner methodology, narrower coverage). The panel coalesces them, preferring
all-transactions and filling gaps from purchase-only, which is what gives
`index_sa` its near-complete metro coverage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cache import fetch
from ..config import RAW_DIR
from .base import Source

FHFA_BASE = "https://www.fhfa.gov/hpi/download/quarterly_datasets"

# The all-transactions metro file ships without a header row.
AT_COLUMNS = ["metro_name", "cbsa", "year", "qtr", "index_nsa", "index_sa"]


class HousePriceIndex(Source):
    name = "hpi"
    description = "FHFA House Price Index by metro (all-transactions + purchase-only)"
    value_columns = ["index_sa", "metro_name"]

    def fetch(self, *, refresh: bool = False) -> Path:
        self._at_path = fetch(
            f"{FHFA_BASE}/hpi_at_metro.csv",
            "fhfa_hpi_at_metro.csv",
            refresh=refresh,
            fallback=RAW_DIR / "hpi_at_metro.csv",
        )
        # Purchase-only is distributed as a spreadsheet at an unstable path, so
        # the committed copy is the primary route for it.
        self._po_path = RAW_DIR / "hpi_po_metro.csv"
        return self._at_path

    def normalize(self, path: Path) -> pd.DataFrame:
        all_tx = self._read_all_transactions(self._at_path)
        purchase_only = self._read_purchase_only(self._po_path)

        merged = all_tx.merge(
            purchase_only, on=["cbsa", "year", "qtr"], how="outer", suffixes=("", "_po")
        )
        # Prefer all-transactions; fall back to purchase-only where it is missing.
        merged["index_sa"] = merged["index_at"].combine_first(merged.get("index_po"))
        merged = merged.dropna(subset=["index_sa"])

        return merged[["cbsa", "year", "qtr", "index_sa", "metro_name"]].drop_duplicates(
            subset=["cbsa", "year", "qtr"], keep="first"
        )

    @staticmethod
    def _read_all_transactions(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, header=None, names=AT_COLUMNS, dtype=str)
        # Some vintages include a header row; drop it if present.
        raw = raw[raw["cbsa"].str.strip().str.lower() != "area code"]
        frame = pd.DataFrame(
            {
                "cbsa": pd.to_numeric(raw["cbsa"], errors="coerce"),
                "year": pd.to_numeric(raw["year"], errors="coerce"),
                "qtr": pd.to_numeric(raw["qtr"], errors="coerce"),
                # Values use "-" for missing and may carry parentheses.
                "index_at": pd.to_numeric(
                    raw["index_sa"].str.replace(r"[()\s$,]", "", regex=True),
                    errors="coerce",
                ),
                "metro_name": raw["metro_name"].str.strip(),
            }
        ).dropna(subset=["cbsa", "year", "qtr"])
        frame[["cbsa", "year", "qtr"]] = frame[["cbsa", "year", "qtr"]].astype(int)
        return frame.drop_duplicates(subset=["cbsa", "year", "qtr"], keep="first")

    @staticmethod
    def _read_purchase_only(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["cbsa", "year", "qtr", "index_po"])
        raw = pd.read_csv(path, dtype=str)
        raw.columns = [c.strip().lower() for c in raw.columns]

        year_col = next((c for c in raw.columns if c in ("yr", "year")), None)
        index_col = next((c for c in raw.columns if "index_sa" in c), None)
        if year_col is None or index_col is None or "cbsa" not in raw.columns:
            return pd.DataFrame(columns=["cbsa", "year", "qtr", "index_po"])

        frame = pd.DataFrame(
            {
                "cbsa": pd.to_numeric(raw["cbsa"], errors="coerce"),
                "year": pd.to_numeric(raw[year_col], errors="coerce"),
                "qtr": pd.to_numeric(raw["qtr"], errors="coerce"),
                "index_po": pd.to_numeric(raw[index_col], errors="coerce"),
            }
        ).dropna(subset=["cbsa", "year", "qtr"])
        frame[["cbsa", "year", "qtr"]] = frame[["cbsa", "year", "qtr"]].astype(int)
        return frame.drop_duplicates(subset=["cbsa", "year", "qtr"], keep="first")
