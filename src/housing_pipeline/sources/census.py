"""Census Bureau sources: metro population estimates and ACS median income.

The population table doubles as this pipeline's naming authority -- it is what
every free-text metro label gets matched against -- so it is also the source
that has to get Metropolitan Divisions right.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from ..cache import cache_path, fetch
from ..cbsa import apply_income_aliases, to_quarterly
from ..config import ACS_YEARS, RAW_DIR, USER_AGENT, census_api_key
from .base import Source

log = logging.getLogger(__name__)

POPEST_BASE = "https://www2.census.gov/programs-surveys/popest/datasets"

# Two vintages are needed for full historical coverage; OMB revised metro
# definitions between them, so codes present in one may be absent from the other.
POP_VINTAGES = [
    (
        f"{POPEST_BASE}/2010-2019/metro/totals/cbsa-est2019-alldata.csv",
        "cbsa-est2019-alldata.csv",
        "cbsa-est2019-alldata.csv",
    ),
    (
        f"{POPEST_BASE}/2020-2024/metro/totals/cbsa-est2024-alldata.csv",
        "cbsa-est2024-alldata.csv",
        "cbsa-est2025-alldata.csv",
    ),
]


class Population(Source):
    name = "population"
    description = "Census metro population estimates (also the metro-naming authority)"
    value_columns = ["population", "metro_name"]

    def fetch(self, *, refresh: bool = False) -> Path:
        # Multi-file source: fetch each vintage, return the cache directory.
        self._paths = []
        for url, cache_name, fallback_name in POP_VINTAGES:
            self._paths.append(
                fetch(
                    url,
                    cache_name,
                    refresh=refresh,
                    fallback=RAW_DIR / fallback_name,
                )
            )
        return self._paths[0].parent

    def normalize(self, path: Path) -> pd.DataFrame:
        frames = [self._read_vintage(p) for p in self._paths]
        stacked = pd.concat(frames, ignore_index=True)
        # Later vintages win where they overlap.
        stacked = stacked.drop_duplicates(subset=["cbsa", "year"], keep="last")
        annual = stacked.sort_values(["cbsa", "year"]).reset_index(drop=True)
        return to_quarterly(annual)

    @staticmethod
    def _read_vintage(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, encoding="iso-8859-1")

        # Metropolitan Divisions carry their own code in MDIV, and FHFA reports
        # large multi-division metros (Chicago, NYC, LA, Tampa, ...) under those
        # codes. Keeping only LSAD == "Metropolitan Statistical Area" and reading
        # CBSA -- the previous behavior -- silently dropped 37 metros' population.
        keep = raw["LSAD"].isin(["Metropolitan Statistical Area", "Metropolitan Division"])
        rows = raw[keep].copy()
        is_division = rows["LSAD"].eq("Metropolitan Division")
        rows["cbsa"] = rows["MDIV"].where(is_division, rows["CBSA"])
        rows = rows.dropna(subset=["cbsa"])
        rows["cbsa"] = rows["cbsa"].astype(int)

        pop_cols = [c for c in rows.columns if c.startswith("POPESTIMATE")]
        long = rows.melt(
            id_vars=["cbsa", "NAME"],
            value_vars=pop_cols,
            var_name="year",
            value_name="population",
        ).rename(columns={"NAME": "metro_name"})
        long["year"] = long["year"].str.extract(r"(\d{4})").astype(int)
        return long.dropna(subset=["population"])


class ACSIncome(Source):
    """ACS 5-year median household income (table B19013) by metro.

    Requires a free Census API key in CENSUS_API_KEY. This is the one source
    that cannot fall back to a committed file, because income is what defines
    the affordability target -- a stale copy would silently change the labels.
    """

    name = "acs_income"
    description = "ACS 5-year median household income by metro (defines the target)"
    value_columns = ["median_income"]

    GEO = "metropolitan statistical area/micropolitan statistical area:*"
    VARIABLE = "B19013_001E"

    def fetch(self, *, refresh: bool = False) -> Path:
        dest = cache_path("acs_median_income.csv")
        if dest.exists() and not refresh:
            return dest

        key = census_api_key()
        if key is None:
            raise RuntimeError(
                "CENSUS_API_KEY is not set. Get a free key at "
                "https://api.census.gov/data/key_signup.html and export it, "
                "then re-run. (ACS income defines the affordability target, so "
                "the pipeline will not substitute a stale copy.)"
            )

        rows = []
        for year in ACS_YEARS:
            try:
                rows.append(self._fetch_year(year, key))
            except Exception as exc:  # noqa: BLE001
                # Not every ACS vintage is published for every year; skipping a
                # missing one is normal and should not abort the whole build.
                log.warning("ACS %s unavailable (%s); skipping", year, exc)

        if not rows:
            raise RuntimeError("no ACS years could be retrieved")

        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(dest, index=False)
        return dest

    def _fetch_year(self, year: int, key: str) -> pd.DataFrame:
        params = urllib.parse.urlencode(
            {"get": f"NAME,{self.VARIABLE}", "for": self.GEO, "key": key}
        )
        url = f"https://api.census.gov/data/{year}/acs/acs5?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)

        header, *body = payload
        frame = pd.DataFrame(body, columns=header)
        frame = frame.rename(
            columns={
                self.VARIABLE: "median_income",
                header[-1]: "cbsa",
            }
        )
        frame["year"] = year
        frame["cbsa"] = pd.to_numeric(frame["cbsa"], errors="coerce")
        frame["median_income"] = pd.to_numeric(frame["median_income"], errors="coerce")
        # ACS uses large negative sentinels for suppressed values.
        frame.loc[frame["median_income"] < 0, "median_income"] = pd.NA
        return frame[["cbsa", "year", "median_income"]].dropna(subset=["cbsa"])

    def normalize(self, path: Path) -> pd.DataFrame:
        income = pd.read_csv(path)
        income["cbsa"] = income["cbsa"].astype(int)
        income = income.dropna(subset=["median_income"])
        income = apply_income_aliases(income)
        income = income.drop_duplicates(subset=["cbsa", "year"], keep="first")
        return to_quarterly(income[["cbsa", "year", "median_income"]])
