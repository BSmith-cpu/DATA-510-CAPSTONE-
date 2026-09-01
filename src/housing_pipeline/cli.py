"""Command line interface.

    python -m housing_pipeline build           # fetch, join, label, engineer
    python -m housing_pipeline build --refresh # ignore cache, re-download
    python -m housing_pipeline info            # what is in the built panel
    python -m housing_pipeline sources         # list sources and coverage
    python -m housing_pipeline clear-cache
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import FEATURES_PATH, PANEL_PATH, census_api_key
from .features import build_features, feature_coverage
from .panel import build_panel, load_panel, save_panel
from .sources import SOURCE_NAMES, build_sources


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )


def cmd_build(args: argparse.Namespace) -> int:
    skip = set(args.skip or [])
    panel = build_panel(refresh=args.refresh, skip=skip)

    print("\nSource coverage")
    print("-" * 64)
    for name, summary in panel.attrs.get("source_report", {}).items():
        print(f"  {name:<14} {summary}")

    save_panel(panel, PANEL_PATH)

    features = build_features(panel)
    features.to_parquet(FEATURES_PATH, index=False)

    print("\nPanel")
    print("-" * 64)
    print(f"  rows                    {len(panel):,}")
    print(f"  metros                  {panel['cbsa'].nunique():,}")
    print(f"  quarters                {panel['year'].min()}Q{panel.loc[panel['year'].idxmin(), 'qtr']}"
          f" - {panel['year'].max()}Q{panel.loc[panel['year'].idxmax(), 'qtr']}")
    print(f"  confirmed onsets        {int(panel['collapse_onset_confirmed'].sum()):,}")
    print(f"  unconfirmed onsets      "
          f"{int(panel['collapse_onset'].sum() - panel['collapse_onset_confirmed'].sum()):,}")

    print("\nFeature coverage (metros with a usable value)")
    print("-" * 64)
    for _, row in feature_coverage(features).iterrows():
        print(f"  {row['feature']:<38} {row['metros']:>4} metros  {row['rows']:>7,} rows")

    print(f"\nWrote {PANEL_PATH}")
    print(f"Wrote {FEATURES_PATH}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        panel = load_panel()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Panel: {PANEL_PATH}")
    print(f"  rows            {len(panel):,}")
    print(f"  metros          {panel['cbsa'].nunique():,}")
    print(f"  years           {panel['year'].min()} - {panel['year'].max()}")
    print(f"  columns         {len(panel.columns)}")
    print("\nNon-null coverage by column (metros):")
    for column in panel.columns:
        if column in ("cbsa", "year", "qtr"):
            continue
        metros = panel.loc[panel[column].notna(), "cbsa"].nunique()
        print(f"  {column:<32} {metros:>4}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    sources = build_sources()
    print(f"{'source':<14} {'national':<9} description")
    print("-" * 78)
    for name in SOURCE_NAMES:
        source = sources[name]
        flag = "yes" if source.national else ""
        print(f"{name:<14} {flag:<9} {source.description}")

    if census_api_key() is None:
        print(
            "\nNote: CENSUS_API_KEY is not set, so 'acs_income' cannot be built.\n"
            "      Free key: https://api.census.gov/data/key_signup.html"
        )
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    from .cache import clear

    clear(args.name)
    print("Cleared cache" + (f" entry {args.name}" if args.name else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="housing_pipeline",
        description="Build the metro housing affordability panel.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="fetch, join, label, and engineer features")
    build.add_argument(
        "--refresh", action="store_true", help="ignore the cache and re-download"
    )
    build.add_argument(
        "--skip", nargs="*", choices=SOURCE_NAMES, help="skip these sources"
    )
    build.set_defaults(func=cmd_build)

    info = sub.add_parser("info", help="summarize the built panel")
    info.set_defaults(func=cmd_info)

    sources = sub.add_parser("sources", help="list available sources")
    sources.set_defaults(func=cmd_sources)

    clear_cache = sub.add_parser("clear-cache", help="delete cached downloads")
    clear_cache.add_argument("name", nargs="?", help="one cached filename, or all")
    clear_cache.set_defaults(func=cmd_clear_cache)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    pd.set_option("display.width", 120)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
