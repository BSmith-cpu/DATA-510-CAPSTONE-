"""The contract every data source implements.

A source is responsible for exactly two things: getting its bytes onto local
disk, and reshaping them into the panel's key. Everything after that -- joining,
feature engineering, modeling -- is written once and works for any source that
honors this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# Every normalized frame is keyed on these columns.
PANEL_KEYS = ["cbsa", "year", "qtr"]


class Source(ABC):
    """A single upstream dataset.

    Attributes
    ----------
    name:
        Short identifier used on the command line and in logs.
    description:
        One line explaining what the source contributes, shown by `list-sources`.
    value_columns:
        Columns this source adds to the panel, excluding the key columns.
    national:
        True for series that carry no metro dimension (currently only the S&P
        500). These join on year/quarter alone and are broadcast to every metro.
    """

    name: str
    description: str
    value_columns: list[str]
    national: bool = False

    @abstractmethod
    def fetch(self, *, refresh: bool = False) -> Path:
        """Download (or locate) this source's raw bytes and return the path."""

    @abstractmethod
    def normalize(self, path: Path) -> pd.DataFrame:
        """Reshape raw bytes into a frame keyed on cbsa/year/qtr (or year/qtr)."""

    def load(self, *, refresh: bool = False) -> pd.DataFrame:
        """Fetch then normalize, validating the result against the contract."""
        frame = self.normalize(self.fetch(refresh=refresh))
        self._validate(frame)
        return frame

    def _validate(self, frame: pd.DataFrame) -> None:
        expected_keys = ["year", "qtr"] if self.national else PANEL_KEYS
        missing = [c for c in expected_keys if c not in frame.columns]
        if missing:
            raise ValueError(f"source '{self.name}' is missing key column(s): {missing}")

        missing_values = [c for c in self.value_columns if c not in frame.columns]
        if missing_values:
            raise ValueError(
                f"source '{self.name}' declares value columns it did not "
                f"produce: {missing_values}"
            )

        duplicated = frame.duplicated(subset=expected_keys).sum()
        if duplicated:
            raise ValueError(
                f"source '{self.name}' produced {duplicated} duplicate rows for "
                f"its key {expected_keys}; the panel join would fan out"
            )

    def coverage(self, frame: pd.DataFrame) -> str:
        """Human-readable coverage summary, used by the build report."""
        if self.national:
            return f"{len(frame)} quarters (national)"
        return f"{frame['cbsa'].nunique()} metros, {len(frame)} rows"


def quarterly_mean(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Collapse a monthly series to quarterly means.

    Shared by every Zillow product, which all ship the same wide monthly layout.
    """
    working = frame.copy()
    working["year"] = working[date_col].dt.year
    working["qtr"] = working[date_col].dt.quarter
    grouped = (
        working.groupby([*group_cols, "year", "qtr"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: out_col})
    )
    return grouped
