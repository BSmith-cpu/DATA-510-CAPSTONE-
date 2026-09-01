# Where Next? Predicting the Next Unaffordable U.S. City

An early-warning model that ranks U.S. metros by their risk of tipping into a housing affordability crisis, built on nine public data sources.

**Author:** Brandon Smith

## Quick start

```bash
pip install -e ".[model,dev]"        # install the pipeline
export CENSUS_API_KEY=...            # free: api.census.gov/data/key_signup.html
python -m housing_pipeline build     # fetch, join, label, engineer features
```

That writes `data/build/panel.parquet` and `data/build/features.parquet`. Both
are committed, so if you only want to work with the data you can skip the build
entirely:

```python
from housing_pipeline import load_panel, build_features, FEATURE_NAMES

panel = load_panel()                 # joined + labelled, 410 metros
frame = build_features(panel)        # + 15 leakage-safe model features
```

Other commands:

| Command | Purpose |
|---|---|
| `python -m housing_pipeline sources` | List the nine sources and what each contributes |
| `python -m housing_pipeline check` | Report source freshness without rebuilding |
| `python -m housing_pipeline info` | Column-by-column coverage of the built panel |
| `python -m housing_pipeline build --refresh` | Ignore the cache and re-download |
| `python -m housing_pipeline build --skip wages` | Build without a slow or unavailable source |
| `python -m housing_pipeline clear-cache` | Drop cached downloads |
| `python -m housing_pipeline build --fail-on-stale` | Rebuild, exiting non-zero if a source is behind (for scheduled runs) |
| `pytest` | Run the test suite |

## Repo contents

| Path | Purpose |
|------|---------|
| [`src/housing_pipeline/`](src/housing_pipeline/) | The data pipeline: sources, joining, labelling, features |
| [`tests/`](tests/) | Tests for CBSA matching, leakage safety, and the panel join |
| [`notebooks/`](notebooks/) | Modeling and reporting notebooks (they consume the pipeline) |
| [`data/build/`](data/build/) | Built panel and feature table (Parquet, committed) |
| [`data/cache/`](data/) | Raw upstream downloads — gitignored, rebuilt on demand |
| [`deliverables/`](deliverables/) | Milestone write-ups and the post-submission methodology note |

## Data sources

All nine are public. Everything except the ACS pull works without credentials.

| Source | Contributes | Notes |
|---|---|---|
| Census Population Estimates | Population, metro names | Also the naming authority for metro matching |
| FHFA House Price Index | `index_sa` | All-transactions, filled from purchase-only |
| Census ACS (B19013) | Median household income | Needs `CENSUS_API_KEY`; defines the target |
| Zillow ZHVI | Home values | |
| Zillow ZORI | Rents | |
| Zillow For-Sale Inventory | Housing supply | Starts 2018 |
| BLS LAUS | Unemployment rate | |
| BLS QCEW | Wages, employment | ~300MB/year, cached after first fetch |
| FRED | S&P 500 | The one national series; broadcast to all metros |

## How it works

`build_panel()` loads the population table first — it is both the metro-naming
authority and the CBSA universe that two other sources need — then loads every
remaining source and left-joins them onto the FHFA spine. Each source declares
its own key and value columns, and the join **raises rather than silently
duplicating rows** if a source's key is not unique.

Adding a tenth source means writing one class in `src/housing_pipeline/sources/`
and listing it in the registry. Nothing downstream changes.

### The target

Three labels, increasingly strict:

ACS income is annual, so it is **interpolated** across quarters rather than
step-broadcast. Repeating one figure for four quarters put a sawtooth into
`price_to_income_ratio` — the model's most important feature — which dropped
~1.9% every Q1 and climbed Q2–Q4. That artifact suppressed Q1 onsets entirely
and produced a spurious accuracy dip at exactly a 4-quarter lookahead.

- `is_unaffordable` — price-to-income above 5.0. A persistent *state*. Kept for
  reference only: it changes about 4% of the time year over year, so a "nothing
  changed" baseline beats a trained model on it.
- `collapse_onset` — the first quarter a metro crosses the threshold.
- **`collapse_onset_confirmed`** — an onset that still holds the next quarter.
  This is the modeling target. It excludes single-quarter reversions and
  crossings at the edge of the data window that cannot be verified yet.

### Leakage safety

Every predictor is lagged four quarters, applied *after* any percent-change or
rolling calculation. `tests/test_features.py` asserts this directly by
perturbing recent quarters and checking that earlier feature values do not move.

Features also declare whether their direction is knowable. Ten carry a monotonic
constraint into the model; five (unemployment, inventory, S&P 500, and both
wage/employment series) are left unconstrained because their sign is genuinely
arguable.

## Current results

Grouped cross-validation, cities never split across folds:

| Metric | Value | No-skill baseline |
|---|---|---|
| PR-AUC | 0.370 | 0.036 |
| ROC-AUC | 0.915 | 0.500 |

Walk-forward (temporal) backtest, refitting each quarter on only what was known
then — the honest number for an early-warning claim:

| Metric | Value | No-skill baseline |
|---|---|---|
| PR-AUC | 0.053 | 0.010 |
| ROC-AUC | 0.866 | 0.500 |

A 20-metro watchlist would have contained **79%** of onsets in the quarter they
occurred and **71%** one quarter ahead, falling to ~44% at two quarters and
toward noise beyond three. Useful as a 1–2 quarter lookahead; not longer.

Trained on 84 metros with a confirmed collapse event or a near-miss (a 4.5–5.0
ratio that never crossed, serving as a hard negative). 262 at-risk metros are
scored for the watchlist.

### Reading the watchlist

The model trains on a population roughly **4× denser in events** than the one it
scores (4.65% vs 1.17%), because confirmed onsets are too sparse to learn from
otherwise. Uncorrected, that made the output unusable: the tuned threshold sat
at 0.31 while the top-scoring metro scored 0.16, so the alert could never fire.

`housing_pipeline.scoring` produces two things from the raw model output:

- **`risk_rank` / `risk_percentile` / `risk_tier`** — relative standing among
  the scored metros. Always valid, because rank ordering is what the grouped
  cross-validation actually measured. **This is the primary product.**
- **`risk_probability`** — the raw score shifted from the training prior onto
  the deployment prior, anchored to a base rate measured from observed data
  rather than assumed. Correction is monotonic, so it never reorders the list.

Current tiers: 14 Elevated, 39 Watch, 209 Monitor.

For the full limitations — precision at the operating point, the 1–2 year ACS
income lag that bounds how early any warning can be, and the fairness review
that has not yet been run — see
[`deliverables/POST-SUBMISSION-METHODOLOGY-UPDATE.md`](deliverables/POST-SUBMISSION-METHODOLOGY-UPDATE.md).

## Notebooks

- [`notebooks/w04-expoloration-housing/`](notebooks/w04-expoloration-housing/) —
  original EDA. The R data-assembly logic it contained has been superseded by
  `housing_pipeline`; the notebook is retained for its exploratory analysis.
- [`notebooks/w07-data-summary/`](notebooks/w07-data-summary/) — modeling,
  validation, SHAP, and holdout scoring. Consumes the pipeline directly.
