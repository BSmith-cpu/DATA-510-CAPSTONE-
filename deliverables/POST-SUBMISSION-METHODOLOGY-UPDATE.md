# Post-Submission Methodology Update

**Applies to:** `M14-Final Poster/BSmith Poster Final.pdf` and `M15-Final Writeup/writeup.pdf`

Both of those PDFs are finished, compiled deliverables with no editable source committed to this repo, so they haven't been (and can't easily be) edited in place. This document records what changed in the modeling work after they were submitted, so anyone reading the final poster/write-up alongside the current notebooks understands the numbers no longer match -- and why the new ones are more trustworthy, not just different.

## What the submitted deliverables describe

- 3 training cities (Austin, Boise, Tampa), each hand-picked as a known real-world case.
- A hand-picked 3-4 feature subset per model.
- Target: `is_unaffordable` (a persistent state).
- Reported accuracy/F1 as the headline metrics.

## What changed, and why

1. **Coverage and join bugs fixed.** Population, ZHVI, and income-join bugs meant the pipeline was silently limited to far fewer metros than the source data actually supported. Fixed and verified (e.g. population coverage 373/410 -> 410/410; median income going from effectively 3 cities to 374).
2. **Three new real data sources added:** BLS local unemployment rate, Zillow for-sale inventory, and the S&P 500 -- fetched live, matched to CBSA, lagged the same way as every other feature.
3. **A real data-leakage bug was found and fixed.** The internal train/validation split let the same city appear on both sides (confirmed empirically: 19/19 overlapping cities). Fixed with `StratifiedGroupKFold` throughout, including the hyperparameter-tuning loop.
4. **The target itself was found to be the wrong one.** `is_unaffordable` only changes 3.9% of the time per year, so a trivial "assume nothing changed" baseline scored *higher* (accuracy 0.961, F1 0.852) than the fitted model -- the impressive-looking old numbers were mostly an artifact of a slow-moving label, not genuine early-warning skill.
5. **Target switched to `collapse_onset`** (the actual transition event, restricted to metros not already unaffordable) -- a target that cannot be gamed by persistence (a no-skill baseline scores F1 = 0.0 on it).
6. **Training population rebuilt around real events, not geography.** The old 3-city population gave almost no usable positive examples for the honest target. The new population is every city (57) with complete feature data that has had a real `collapse_onset` event -- selected because it has the data needed, not because it represents a US region.
7. **Metrics reporting fixed to match a rare-event problem.** Accuracy/F1 alone are misleading at a ~10% positive rate; PR-AUC (average precision) is now the primary reported metric, always shown next to the no-skill baseline.

## Current headline results (see `notebooks/w07-data-summary/output/tables/` for the full, regenerated tables)

- **Model 2 (deployed scorer):** PR-AUC 0.451 (validation) / 0.587 (grouped 10-fold CV) vs. a 0.099 no-skill baseline -- roughly a 6x improvement over chance. ROC-AUC 0.910 (validation) / 0.914 (CV).
- **Diagnostic check:** removing the price-to-income features (the ones closest to a persistence proxy) still gives PR-AUC 0.543 / AUC 0.844 -- confirms the model isn't just riding on one dominant feature.
- **Feature importance (SHAP), most to least important:** `price_to_income_lag`, `zhvi_qoq_lag`, `unemployment_rate_lag`, `three-year_home_price_growth_trend`, `zhvi_yoy_lag`, `zori_yoy_lag`, `sp500_yoy_lag`, `pop_velocity_lag`, `pop_acceleration_lag`, `inv_qoq_lag`, `hpi_3yr_chg_lag`, `hpi_yoy_lag`, `price_to_income_5yr_chg`. Notably more balanced across features than the old target produced (no single feature dominates as heavily), and population growth remains the weakest signal even after every other addition.
- **Leave-one-city-out backtest (57 cities):** mean AUC 0.805, median 0.917. Two known failure cases (Springfield, MA and Traverse City, MI, both AUC 0.0) were investigated and traced to a genuine data limitation, not a modeling flaw: both cities' only `collapse_onset` event crossed the 5.0 threshold by a razor-thin margin (0.003-0.025) in the most recent observed quarter, with zero confirmed quarters afterward (ACS income data lags into 2025, so the label can't yet be verified as a real, sustained collapse the way Austin/Boise/Tampa's multi-year-confirmed onsets can). All 8 of the training population's 2024 onset events share this same thin-margin, recently-unconfirmed pattern -- worth treating as lower-confidence labels in any future refinement.
- **Holdout scoring:** 296 at-risk metros currently scored for early-warning risk (see `holdout_city_risk_scores.csv` for the full ranking).

## Where to find the current work

- `notebooks/w04-expoloration-housing/W04-exploration-housing.rmd` -- data join pipeline, including the new data sources.
- `notebooks/w07-data-summary/w07-data-summary.ipynb` -- feature engineering, target definition, training population selection, both models, validation, SHAP, and holdout scoring.
- `notebooks/w07-data-summary/output/` -- all regenerated metrics tables and figures.
