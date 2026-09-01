"""Assemble every source into one quarterly metro panel, then label it.

This replaces the R notebook that previously did the joining. The whole build is
one function call, and it reports what each source contributed instead of
leaving coverage to be discovered later by a surprised analyst.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import AFFORDABILITY_THRESHOLD, PANEL_PATH, ensure_dirs
from .sources import SOURCE_NAMES, build_sources

log = logging.getLogger(__name__)


def build_panel(*, refresh: bool = False, skip: set[str] | None = None) -> pd.DataFrame:
    """Build the full quarterly panel with affordability labels attached."""
    ensure_dirs()
    skip = skip or set()

    # Population loads first: it is the naming authority the Zillow matchers need
    # and the CBSA universe the QCEW code expansion needs.
    bootstrap = build_sources()
    population = bootstrap["population"].load(refresh=refresh)
    reference = (
        population[["cbsa", "metro_name"]].dropna().drop_duplicates(subset="cbsa")
    )
    known_cbsas = set(reference["cbsa"].astype(int))
    log.info("population: %s", bootstrap["population"].coverage(population))

    sources = build_sources(reference=reference, known_cbsas=known_cbsas)

    frames: dict[str, pd.DataFrame] = {"population": population}
    report: dict[str, str] = {
        "population": bootstrap["population"].coverage(population)
    }

    for name in SOURCE_NAMES:
        if name == "population" or name in skip:
            continue
        source = sources[name]
        try:
            frame = source.load(refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            log.error("source '%s' failed: %s", name, exc)
            report[name] = f"FAILED: {exc}"
            continue
        frames[name] = frame
        report[name] = source.coverage(frame)
        log.info("%s: %s", name, report[name])

    panel = _join(frames, sources)
    panel = add_affordability_labels(panel)
    panel.attrs["source_report"] = report
    return panel


def _join(frames: dict[str, pd.DataFrame], sources) -> pd.DataFrame:
    """Join source frames on the panel key.

    HPI is the spine: it has the broadest metro-quarter coverage and carries the
    metro names FHFA uses. Everything metro-keyed is left-joined onto it;
    national series are broadcast on year/quarter.
    """
    if "hpi" not in frames:
        raise RuntimeError("cannot build a panel without the HPI spine")

    panel = frames["hpi"].copy()

    for name, frame in frames.items():
        if name == "hpi":
            continue

        source = sources[name]
        keys = ["year", "qtr"] if source.national else ["cbsa", "year", "qtr"]
        columns = keys + [c for c in source.value_columns if c in frame.columns]
        incoming = frame[columns].copy()

        # metro_name arrives from both HPI and population; keep the spine's.
        if "metro_name" in incoming.columns and "metro_name" in panel.columns:
            incoming = incoming.drop(columns=["metro_name"])

        before = len(panel)
        panel = panel.merge(incoming, on=keys, how="left")
        if len(panel) != before:
            raise RuntimeError(
                f"join with '{name}' changed the row count "
                f"({before} -> {len(panel)}); its key is not unique"
            )

    return panel.sort_values(["cbsa", "year", "qtr"]).reset_index(drop=True)


def add_affordability_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the price-to-income ratio and the collapse-onset labels.

    Three labels, increasingly strict:

    ``is_unaffordable``
        A persistent state. Kept for reference only -- it is dominated by its own
        autocorrelation (it changes about 4% of the time year over year), so a
        "nothing changed" baseline beats a trained model on it.
    ``collapse_onset``
        The first quarter a metro crosses the threshold. A real transition event.
    ``collapse_onset_confirmed``
        An onset that is still unaffordable the following quarter. This is the
        modeling target: it excludes single-quarter reversions and events at the
        edge of the data window that cannot be verified yet.
    """
    panel = panel.sort_values(["cbsa", "year", "qtr"]).reset_index(drop=True)

    panel["price_to_income_ratio"] = panel["zhvi_qtr"] / panel["median_income"]
    panel["is_unaffordable"] = panel["price_to_income_ratio"] > AFFORDABILITY_THRESHOLD

    grouped = panel.groupby("cbsa")["is_unaffordable"]
    panel["prev_unaffordable"] = grouped.shift(1).fillna(False).astype(bool)
    panel["next_unaffordable"] = grouped.shift(-1)

    panel["collapse_onset"] = panel["is_unaffordable"] & ~panel["prev_unaffordable"]
    # A null "next quarter" means the window ends here, so the event is
    # unverifiable -- treated the same as an observed reversion.
    panel["collapse_onset_confirmed"] = panel["collapse_onset"] & panel[
        "next_unaffordable"
    ].eq(True)

    return panel


def save_panel(panel: pd.DataFrame, path=PANEL_PATH) -> None:
    ensure_dirs()
    panel.to_parquet(path, index=False)
    log.info("wrote %s (%s rows)", path, len(panel))


def load_panel(path=PANEL_PATH) -> pd.DataFrame:
    """Load the built panel, with a clear error if it has not been built yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it first:\n"
            f"    python -m housing_pipeline build"
        )
    return pd.read_parquet(path)
