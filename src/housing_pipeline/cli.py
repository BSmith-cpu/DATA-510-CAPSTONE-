"""Command line interface.

    python -m housing_pipeline build            # fetch, join, label, engineer
    python -m housing_pipeline build --refresh  # ignore cache, re-download
    python -m housing_pipeline check            # source freshness, no rebuild
    python -m housing_pipeline backtest         # walk-forward temporal backtest
    python -m housing_pipeline info             # what is in the built panel
    python -m housing_pipeline sources          # list sources and coverage
    python -m housing_pipeline clear-cache

For a scheduled refresh, `build --fail-on-stale` exits non-zero when a source
has fallen behind its own release cadence, so a silent upstream outage surfaces
as a failed job rather than a quietly stale watchlist.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import FEATURES_PATH, PANEL_PATH, census_api_key
from .backtest import lead_time, lead_time_summary, walk_forward
from .features import build_features, feature_coverage
from .freshness import check_freshness, compare_builds
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

    # Keep the previous build so structural changes can be reported rather than
    # discovered later by a confused analyst.
    previous = None
    if PANEL_PATH.exists():
        try:
            previous = pd.read_parquet(PANEL_PATH)
        except Exception:  # noqa: BLE001 - a corrupt prior build must not block this one
            previous = None

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

    if previous is not None:
        diff = compare_builds(previous, panel)
        changed = diff[diff["change"].fillna(0) != 0]
        print("\nChange since the previous build")
        print("-" * 64)
        if changed.empty:
            print("  no structural change")
        else:
            for _, row in changed.iterrows():
                pct = "" if pd.isna(row["pct_change"]) else f" ({row['pct_change']:+.1f}%)"
                print(f"  {row['metric']:<32} {row['previous']:>8} -> {row['current']:>8}"
                      f"  {row['change']:+}{pct}")

    report = check_freshness(panel)
    print("\nSource freshness")
    print("-" * 64)
    print(report)

    print(f"\nWrote {PANEL_PATH}")
    print(f"Wrote {FEATURES_PATH}")

    if report.stale and args.fail_on_stale:
        print(f"\nFAILED: stale sources: {', '.join(report.stale)}", file=sys.stderr)
        return 2
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Freshness check against the built panel, for scheduled runs."""
    try:
        panel = load_panel()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    report = check_freshness(panel, tolerance_quarters=args.tolerance)
    print(report)
    if report.stale:
        print(f"\nSTALE: {', '.join(report.stale)}", file=sys.stderr)
        return 2
    print("\nAll sources within their expected publication lag.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward backtest: refit each quarter using only what was known then."""
    try:
        panel = load_panel()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    features = build_features(panel)
    result = walk_forward(
        features,
        start=(args.start_year, args.start_qtr),
        min_train_events=args.min_train_events,
    )

    if result.predictions.empty:
        print("No origin quarters could be evaluated.", file=sys.stderr)
        return 1

    print(result.summary())

    print("\n\nPer-origin (origins containing at least one actual onset)")
    print("-" * 78)
    columns = ["origin", "n_train_events", "n_scored", "n_events",
               "pr_auc", "precision_at_10", "recall_at_20"]
    print(result.evaluable[columns].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    summary = lead_time_summary(lead_time(result, horizon=args.horizon))
    if not summary.empty:
        print("\n\nLead time: rank held before the onset quarter")
        print("-" * 78)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.out:
        result.predictions.to_csv(args.out, index=False)
        print(f"\nWrote per-metro predictions to {args.out}")
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
    build.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="exit non-zero if any source is behind its expected release lag "
             "(intended for scheduled runs)",
    )
    build.set_defaults(func=cmd_build)

    check = sub.add_parser(
        "check", help="report source freshness without rebuilding"
    )
    check.add_argument(
        "--tolerance", type=float, default=2.0,
        help="quarters of slack beyond a source's expected lag (default: 2)",
    )
    check.set_defaults(func=cmd_check)

    backtest = sub.add_parser(
        "backtest",
        help="walk-forward backtest using only data available at each quarter",
    )
    backtest.add_argument("--start-year", type=int, default=2021)
    backtest.add_argument("--start-qtr", type=int, default=1, choices=[1, 2, 3, 4])
    backtest.add_argument(
        "--min-train-events", type=int, default=8,
        help="skip an origin with fewer confirmed events available to train on",
    )
    backtest.add_argument(
        "--horizon", type=int, default=8,
        help="how many quarters before onset to trace ranks back",
    )
    backtest.add_argument("--out", help="write per-metro predictions to this CSV")
    backtest.set_defaults(func=cmd_backtest)

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
