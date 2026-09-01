"""CBSA identity and crosswalk logic.

Every metro-identity bug this project hit lived in this translation layer, so
it is isolated here as pure functions that tests can exercise directly:

* Census publishes Metropolitan Divisions under a separate ``MDIV`` column, and
  FHFA reports large multi-division metros (Chicago, NYC, LA, Tampa, ...) using
  those division codes. Reading only ``CBSA`` silently dropped 37 metros.
* Zillow labels metros with free text ("Austin, TX") while Census uses the full
  title ("Austin-Round Rock-San Marcos, TX"). Joining on the raw strings matched
  almost nothing.
* QCEW encodes a metro as ``C`` plus the first four digits of its CBSA code.
* BLS LAUS encodes it as ``MT<state><cbsa>`` or ``DV<state><division>``.
"""

from __future__ import annotations

import re

import pandas as pd

# Tampa's ACS income is published under the parent MSA (45300) while the rest of
# this panel keys Tampa on the FHFA division code (45294). Income rows are
# duplicated onto the division code so the join lands.
INCOME_CBSA_ALIASES: dict[int, int] = {45300: 45294}


def zillow_name_key(region_name: str) -> tuple[str | None, str | None]:
    """Split a metro label into (first city, first state) for fuzzy matching.

    Zillow says "Austin, TX"; Census says "Austin-Round Rock-San Marcos, TX".
    Truncating at both the first comma and the first hyphen reduces each to
    "Austin". The state abbreviation is kept because a first-city name alone is
    not unique nationally -- there are several Columbus and Portland metros.

    >>> zillow_name_key("Austin-Round Rock-San Marcos, TX")
    ('Austin', 'TX')
    >>> zillow_name_key("New York, NY")
    ('New York', 'NY')
    """
    if not isinstance(region_name, str) or "," not in region_name:
        return (None, None)

    city_part, _, state_part = region_name.partition(",")
    name_short = city_part.split("-")[0].strip()

    state_match = re.match(r"\s*([A-Z]{2})", state_part)
    state_key = state_match.group(1) if state_match else None

    return (name_short or None, state_key)


def build_name_crosswalk(reference: pd.DataFrame) -> pd.DataFrame:
    """Build a (name_short, state_key) -> cbsa lookup from a reference table.

    `reference` needs ``cbsa`` and ``metro_name`` columns -- normally the Census
    population table, which is the authority for metro naming here.
    """
    frame = reference[["cbsa", "metro_name"]].dropna().drop_duplicates().copy()
    keys = frame["metro_name"].map(zillow_name_key)
    frame["name_short"] = [k[0] for k in keys]
    frame["state_key"] = [k[1] for k in keys]
    frame = frame.dropna(subset=["name_short", "state_key"])
    # A few name+state pairs still resolve to more than one CBSA (a metro name
    # that prefixes another in the same state). Keep the first deterministically
    # rather than letting the join fan out into duplicate rows.
    return frame.drop_duplicates(subset=["name_short", "state_key"], keep="first")


def match_metro_names(region_names: pd.Series, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Map free-text metro labels to CBSA codes.

    Returns a frame of ``region_name`` -> ``cbsa`` (nullable Int64), one row per
    distinct input label. Unmatched labels are kept with a null cbsa so callers
    can report coverage honestly instead of silently losing them.
    """
    labels = pd.Series(region_names.dropna().unique(), name="region_name")
    keys = labels.map(zillow_name_key)
    lookup = pd.DataFrame(
        {
            "region_name": labels,
            "name_short": [k[0] for k in keys],
            "state_key": [k[1] for k in keys],
        }
    )
    merged = lookup.merge(
        crosswalk[["name_short", "state_key", "cbsa"]],
        on=["name_short", "state_key"],
        how="left",
    )
    merged["cbsa"] = merged["cbsa"].astype("Int64")
    return merged[["region_name", "cbsa"]].drop_duplicates(subset=["region_name"])


def qcew_area_to_cbsa(area_fips: str, known_cbsas: set[int]) -> int | None:
    """Translate a QCEW ``C####`` metro code into a CBSA code.

    QCEW truncates the five-digit CBSA to four digits behind a ``C``: Austin's
    12420 becomes ``C1242``. Recovering the fifth digit requires the set of CBSA
    codes actually in use, which is why `known_cbsas` is required rather than
    assuming a trailing zero.

    >>> qcew_area_to_cbsa("C1242", {12420})
    12420
    """
    if not isinstance(area_fips, str) or not area_fips.startswith("C"):
        return None
    prefix = area_fips[1:]
    if len(prefix) != 4 or not prefix.isdigit():
        return None
    matches = [c for c in known_cbsas if str(c).zfill(5).startswith(prefix)]
    # Ambiguity would mean two real metros share a four-digit prefix. That does
    # not occur in the current CBSA universe, and guessing would be worse than
    # dropping the row.
    return matches[0] if len(matches) == 1 else None


def laus_area_to_cbsa(area_code: str) -> int | None:
    """Translate a BLS LAUS area code into a CBSA code.

    LAUS uses ``MT<state-fips><cbsa>`` for metropolitan statistical areas and
    ``DV<state-fips><division>`` for metropolitan divisions. Both carry the
    five-digit code this panel keys on.

    >>> laus_area_to_cbsa("MT4812420000000")
    12420
    """
    if not isinstance(area_code, str):
        return None
    match = re.match(r"^(?:MT|DV)\d{2}(\d{5})", area_code)
    return int(match.group(1)) if match else None


def apply_income_aliases(income: pd.DataFrame) -> pd.DataFrame:
    """Duplicate income rows onto aliased CBSA codes (currently Tampa only).

    ACS publishes Tampa under the parent MSA code while the rest of this panel
    uses the FHFA division code. Rows are duplicated rather than reassigned so
    the parent code keeps working for anything that expects it.
    """
    extra = []
    for source_cbsa, target_cbsa in INCOME_CBSA_ALIASES.items():
        aliased = income[income["cbsa"] == source_cbsa].copy()
        if not aliased.empty:
            aliased["cbsa"] = target_cbsa
            extra.append(aliased)
    if not extra:
        return income
    return pd.concat([income, *extra], ignore_index=True)


def to_quarterly(annual: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """Broadcast an annual table across all four quarters of each year.

    Appropriate for slow-moving stock measures like population, where a step
    function is a fair representation. For anything that feeds a *ratio*, prefer
    `interpolate_annual_to_quarterly` -- see the note there.
    """
    quarters = pd.DataFrame({"qtr": [1, 2, 3, 4]})
    return annual.merge(quarters, how="cross")


def interpolate_annual_to_quarterly(
    annual: pd.DataFrame,
    value_col: str,
    *,
    group_col: str = "cbsa",
    year_col: str = "year",
    anchor_qtr: int = 3,
) -> pd.DataFrame:
    """Spread an annual series smoothly across quarters instead of stepping it.

    Repeating one annual figure across four quarters injects an artificial
    sawtooth into anything derived from it. That is exactly what happened to
    `price_to_income_ratio`, the model's most important feature: with income held
    flat inside a year while home values kept moving, the ratio climbed every
    Q2-Q4 and dropped about 1.9% every Q1 when a new income figure landed. The
    discontinuity was large enough that confirmed onsets never once occurred in
    Q1 -- an artifact of the calendar, not of housing markets.

    Each annual value is anchored at `anchor_qtr` (Q3 by default, since an ACS
    5-year estimate is best read as a mid-period figure) and linearly
    interpolated between anchors. Ends are held flat rather than extrapolated.
    """
    frames = []
    for key, group in annual.groupby(group_col, sort=False):
        group = group.sort_values(year_col)
        years = range(int(group[year_col].min()), int(group[year_col].max()) + 1)
        grid = pd.DataFrame(
            [(y, q) for y in years for q in (1, 2, 3, 4)], columns=[year_col, "qtr"]
        )
        grid[group_col] = key

        anchors = group[[year_col, value_col]].copy()
        anchors["qtr"] = anchor_qtr

        merged = grid.merge(anchors, on=[year_col, "qtr"], how="left")
        merged[value_col] = merged[value_col].interpolate(
            method="linear", limit_direction="both"
        )
        frames.append(merged)

    if not frames:
        return annual.assign(qtr=pd.Series(dtype=int))
    return pd.concat(frames, ignore_index=True)


def month_to_quarter(month: int) -> int:
    """Map a calendar month (1-12) onto its quarter (1-4)."""
    return (int(month) - 1) // 3 + 1
