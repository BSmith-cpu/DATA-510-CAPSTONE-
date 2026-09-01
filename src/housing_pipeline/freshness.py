"""Staleness and drift checks for scheduled rebuilds.

The nine sources update on different clocks -- Zillow monthly, FHFA and QCEW
quarterly, ACS annually -- so a fixed refresh cadence either wastes bandwidth or
serves stale data. These checks report per-source staleness against each one's
own release rhythm, and flag structural changes between builds.

The failure this guards against is quiet: a source silently stops updating, or a
join starts dropping metros, and the watchlist keeps publishing as if nothing
happened. A scheduled run should fail loudly instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# How often each source actually publishes, in quarters, and the column whose
# presence proves the source made it into the panel.
@dataclass(frozen=True)
class SourceCadence:
    column: str
    quarters_between_releases: float
    # Publication lag: how far behind the current quarter a source normally
    # runs even when perfectly healthy. ACS is the extreme case at 1-2 years.
    expected_lag_quarters: float


CADENCES: dict[str, SourceCadence] = {
    "zhvi": SourceCadence("zhvi_qtr", 1 / 3, 1),
    "zori": SourceCadence("zori_qtr", 1 / 3, 1),
    "inventory": SourceCadence("inventory_qtr", 1 / 3, 1),
    "hpi": SourceCadence("index_sa", 1, 2),
    "unemployment": SourceCadence("unemployment_rate", 1 / 3, 1),
    "wages": SourceCadence("qcew_avg_wkly_wage", 1, 3),
    "population": SourceCadence("population", 4, 4),
    "acs_income": SourceCadence("median_income", 4, 8),
    "sp500": SourceCadence("sp500_qtr", 1 / 3, 1),
}


@dataclass
class FreshnessReport:
    rows: list[dict] = field(default_factory=list)

    @property
    def stale(self) -> list[str]:
        return [r["source"] for r in self.rows if r["stale"]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def __str__(self) -> str:
        frame = self.to_frame()
        if frame.empty:
            return "no sources checked"
        lines = [f"{'source':<14} {'latest':<9} {'lag(q)':>7}  {'expected':>8}  status"]
        lines.append("-" * 60)
        for _, row in frame.iterrows():
            status = "STALE" if row["stale"] else "ok"
            latest = f"{row['latest_year']}Q{row['latest_qtr']}" if row["latest_year"] else "MISSING"
            lines.append(
                f"{row['source']:<14} {latest:<9} {row['lag_quarters']:>7.0f}"
                f"  {row['expected_lag']:>8.0f}  {status}"
            )
        return "\n".join(lines)


def _quarter_index(year: int, qtr: int) -> int:
    return int(year) * 4 + int(qtr) - 1


def check_freshness(
    panel: pd.DataFrame,
    *,
    as_of: tuple[int, int] | None = None,
    tolerance_quarters: float = 2.0,
) -> FreshnessReport:
    """Report how far behind each source is relative to its own release rhythm.

    `as_of` defaults to the newest quarter anywhere in the panel. A source is
    stale when its lag exceeds its expected publication lag by more than
    `tolerance_quarters` -- so ACS running two years behind is normal, while
    Zillow running two years behind is not.
    """
    if as_of is None:
        newest = panel.loc[panel["year"].idxmax()]
        latest_year = int(panel["year"].max())
        latest_qtr = int(panel.loc[panel["year"] == latest_year, "qtr"].max())
        as_of = (latest_year, latest_qtr)

    now = _quarter_index(*as_of)
    report = FreshnessReport()

    for name, cadence in CADENCES.items():
        if cadence.column not in panel.columns:
            report.rows.append(
                {
                    "source": name, "column": cadence.column,
                    "latest_year": None, "latest_qtr": None,
                    "lag_quarters": float("inf"), "expected_lag": cadence.expected_lag_quarters,
                    "metros": 0, "stale": True,
                }
            )
            continue

        present = panel[panel[cadence.column].notna()]
        if present.empty:
            report.rows.append(
                {
                    "source": name, "column": cadence.column,
                    "latest_year": None, "latest_qtr": None,
                    "lag_quarters": float("inf"), "expected_lag": cadence.expected_lag_quarters,
                    "metros": 0, "stale": True,
                }
            )
            continue

        year = int(present["year"].max())
        qtr = int(present.loc[present["year"] == year, "qtr"].max())
        lag = now - _quarter_index(year, qtr)

        report.rows.append(
            {
                "source": name,
                "column": cadence.column,
                "latest_year": year,
                "latest_qtr": qtr,
                "lag_quarters": lag,
                "expected_lag": cadence.expected_lag_quarters,
                "metros": int(present["cbsa"].nunique()),
                "stale": lag > cadence.expected_lag_quarters + tolerance_quarters,
            }
        )

    return report


def compare_builds(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Structural diff between two builds of the panel.

    Catches the quiet failures: metros silently dropping out of a join, a
    source's coverage collapsing, or the confirmed-onset count moving sharply
    without an obvious cause.
    """
    rows = [
        {
            "metric": "rows",
            "previous": len(previous),
            "current": len(current),
        },
        {
            "metric": "metros",
            "previous": previous["cbsa"].nunique(),
            "current": current["cbsa"].nunique(),
        },
    ]

    if "collapse_onset_confirmed" in previous and "collapse_onset_confirmed" in current:
        rows.append(
            {
                "metric": "confirmed_onsets",
                "previous": int(previous["collapse_onset_confirmed"].sum()),
                "current": int(current["collapse_onset_confirmed"].sum()),
            }
        )

    for cadence in CADENCES.values():
        column = cadence.column
        if column in previous.columns and column in current.columns:
            rows.append(
                {
                    "metric": f"metros_with_{column}",
                    "previous": int(previous.loc[previous[column].notna(), "cbsa"].nunique()),
                    "current": int(current.loc[current[column].notna(), "cbsa"].nunique()),
                }
            )

    frame = pd.DataFrame(rows)
    frame["change"] = frame["current"] - frame["previous"]
    frame["pct_change"] = (
        frame["change"] / frame["previous"].replace(0, pd.NA) * 100
    ).round(2)
    return frame


def compare_watchlists(
    previous: pd.DataFrame, current: pd.DataFrame, *, top_n: int = 25
) -> pd.DataFrame:
    """Rank movement between two watchlists.

    A metro leaping many positions is either a real signal or a data defect --
    both need a human to look, which is the point of surfacing it.
    """
    prev = previous[["cbsa", "risk_rank"]].rename(columns={"risk_rank": "previous_rank"})
    curr = current[["cbsa", "metro_name", "risk_rank"]].rename(
        columns={"risk_rank": "current_rank"}
    )
    merged = curr.merge(prev, on="cbsa", how="outer")
    merged["rank_change"] = merged["previous_rank"] - merged["current_rank"]

    moved = merged[merged["current_rank"].le(top_n) | merged["previous_rank"].le(top_n)]
    return moved.sort_values("rank_change", ascending=False, na_position="last")
