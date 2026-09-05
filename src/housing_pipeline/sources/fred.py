"""FRED macro series.

The market index is the one national series in the panel: identical for every
metro in a given quarter. It is flagged `national=True` so the panel broadcasts
it on year/quarter rather than joining on a metro key it does not have.

**Why NASDAQ Composite and not the S&P 500.** FRED redistributes the S&P 500
under a licence that caps history at ten years, so its series began in 2016 and
silently truncated the panel -- the derived feature could not exist before
2018Q3. The NASDAQ Composite carries the same broad-market signal with history
back to 1971, which is what a backtest reaching into the 2010s needs. The column
is named for what it is rather than inheriting the old `sp500` label.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cache import fetch
from ..config import RAW_DIR
from .base import Source


class MarketIndex(Source):
    name = "market"
    description = "Broad market index from FRED (national; broadcast to every metro)"
    value_columns = ["market_qtr"]
    national = True

    # NASDAQ Composite: daily close, 1971-present, no licence-imposed truncation.
    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM"

    def fetch(self, *, refresh: bool = False) -> Path:
        return fetch(
            self.URL,
            "fred_market.csv",
            refresh=refresh,
            fallback=RAW_DIR / "sp500_raw.csv",
        )

    def normalize(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        date_col, value_col = raw.columns[0], raw.columns[1]

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col], errors="coerce"),
                # FRED writes "." for non-trading days.
                "market": pd.to_numeric(raw[value_col], errors="coerce"),
            }
        ).dropna()

        frame["year"] = frame["date"].dt.year
        frame["qtr"] = frame["date"].dt.quarter
        return (
            frame.groupby(["year", "qtr"], as_index=False)["market"]
            .mean()
            .rename(columns={"market": "market_qtr"})
        )
