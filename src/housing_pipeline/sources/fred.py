"""FRED macro series.

The S&P 500 is the one national series in the panel: identical for every metro
in a given quarter. It is flagged `national=True` so the panel broadcasts it on
year/quarter rather than trying to join it on a metro key it does not have.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cache import fetch
from ..config import RAW_DIR
from .base import Source


class SP500(Source):
    name = "sp500"
    description = "S&P 500 index from FRED (national; broadcast to every metro)"
    value_columns = ["sp500_qtr"]
    national = True

    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"

    def fetch(self, *, refresh: bool = False) -> Path:
        return fetch(
            self.URL,
            "fred_sp500.csv",
            refresh=refresh,
            fallback=RAW_DIR / "sp500_raw.csv",
        )

    def normalize(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        date_col, value_col = raw.columns[0], raw.columns[1]

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col], errors="coerce"),
                "sp500": pd.to_numeric(raw[value_col], errors="coerce"),
            }
        ).dropna()

        frame["year"] = frame["date"].dt.year
        frame["qtr"] = frame["date"].dt.quarter
        return (
            frame.groupby(["year", "qtr"], as_index=False)["sp500"]
            .mean()
            .rename(columns={"sp500": "sp500_qtr"})
        )
