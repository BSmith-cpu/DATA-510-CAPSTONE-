"""Source registry.

Adding a dataset to the panel means writing one class here and listing it in
`build_sources`. Nothing downstream needs to change.
"""

from __future__ import annotations

import pandas as pd

from .base import PANEL_KEYS, Source
from .bls import Unemployment, Wages
from .census import ACSIncome, Population
from .fhfa import HousePriceIndex
from .fred import SP500
from .zillow import ZHVI, ZORI, Inventory

__all__ = [
    "PANEL_KEYS",
    "Source",
    "Population",
    "ACSIncome",
    "HousePriceIndex",
    "ZHVI",
    "ZORI",
    "Inventory",
    "Unemployment",
    "Wages",
    "SP500",
    "build_sources",
    "SOURCE_NAMES",
]

# Order matters only for readability in the build report; the panel join is
# driven explicitly in panel.py.
SOURCE_NAMES = [
    "population",
    "hpi",
    "acs_income",
    "zhvi",
    "zori",
    "inventory",
    "unemployment",
    "wages",
    "sp500",
]


def build_sources(
    reference: pd.DataFrame | None = None,
    known_cbsas: set[int] | None = None,
) -> dict[str, Source]:
    """Instantiate every source.

    Two sources need context that only exists after the population table is
    loaded: the Zillow products need the metro-naming reference to match their
    free-text labels, and QCEW needs the CBSA universe to expand its truncated
    four-digit codes. Passing None yields sources that can still be listed and
    described, just not loaded.
    """
    empty = pd.DataFrame(columns=["cbsa", "metro_name"])
    ref = reference if reference is not None else empty

    return {
        "population": Population(),
        "hpi": HousePriceIndex(),
        "acs_income": ACSIncome(),
        "zhvi": ZHVI(ref),
        "zori": ZORI(ref),
        "inventory": Inventory(ref),
        "unemployment": Unemployment(),
        "wages": Wages(known_cbsas),
        "sp500": SP500(),
    }
