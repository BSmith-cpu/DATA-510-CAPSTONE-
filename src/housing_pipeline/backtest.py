"""Walk-forward backtesting: would this model have warned us in time?

Every validation in this project so far has been cross-sectional. Leave-one-city-out
answers "does this generalize to a metro it has not seen", which is a real
question -- but it trains on the *whole* time period, so a model predicting a
2020 event may have learned from 2023 data. For an early-warning system that is
the wrong test.

This module runs the honest one. At each origin quarter T, the model is refit
using only labels that were observable at T, then scores every at-risk metro for
quarter T. Nothing after T is visible. That is what the system would actually
have produced had it been running.

Two things are measured:

* **Ranking quality per origin** -- PR-AUC and precision/recall at k, since the
  product is a top-N watchlist rather than a threshold alarm.
* **Lead time** -- for metros that did collapse, how highly were they ranked in
  the quarters *before* it happened. This is the early-warning claim tested
  directly, and it is the number a stakeholder actually cares about.

One structural caveat worth knowing when reading results: confirmed onsets never
occur in Q1. ACS income is annual and broadcast across quarters, so the
price-to-income ratio resets each Q1 and crosses mid-year. Origins with no events
cannot contribute a PR-AUC and are reported separately rather than silently
averaged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURE_NAMES, MONOTONE_CONSTRAINTS

TARGET = "collapse_onset_confirmed"


def quarter_index(year, qtr) -> int:
    """Absolute quarter number, so arithmetic across year boundaries is trivial."""
    return np.asarray(year) * 4 + np.asarray(qtr) - 1


def default_model_factory(scale_pos_weight: float):
    """The production model configuration, minus the Optuna search.

    Tuning inside every backtest fold would be both slow and its own source of
    look-ahead, so the backtest uses fixed, defensible hyperparameters.
    """
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=42,
        n_jobs=-1,
    )


@dataclass
class BacktestResult:
    """Everything a walk-forward run produced."""

    predictions: pd.DataFrame          # one row per (origin, metro) scored
    per_origin: pd.DataFrame           # one row per origin quarter
    skipped: list[dict] = field(default_factory=list)

    @property
    def evaluable(self) -> pd.DataFrame:
        """Origins that actually contained at least one event."""
        return self.per_origin[self.per_origin["n_events"] > 0]

    def pooled(self) -> dict:
        """Metrics over all predictions pooled, ignoring origin boundaries."""
        frame = self.predictions
        if frame["actual"].sum() == 0:
            return {}
        return {
            "origins_evaluated": int(self.evaluable.shape[0]),
            "predictions": int(len(frame)),
            "events": int(frame["actual"].sum()),
            "base_rate": float(frame["actual"].mean()),
            "pr_auc": float(average_precision_score(frame["actual"], frame["score"])),
            "roc_auc": float(roc_auc_score(frame["actual"], frame["score"])),
        }

    def summary(self) -> str:
        pooled = self.pooled()
        if not pooled:
            return "no events in any evaluated origin"

        lines = [
            "Walk-forward backtest",
            "-" * 58,
            f"  origins evaluated      {pooled['origins_evaluated']}",
            f"  metro-quarters scored  {pooled['predictions']:,}",
            f"  actual onsets          {pooled['events']}",
            f"  base rate              {pooled['base_rate']:.3%}",
            "",
            f"  pooled PR-AUC          {pooled['pr_auc']:.3f}"
            f"   (no-skill {pooled['base_rate']:.3f})",
            f"  pooled ROC-AUC         {pooled['roc_auc']:.3f}",
        ]

        evaluable = self.evaluable
        if not evaluable.empty:
            lines += [
                "",
                f"  mean precision@10      {evaluable['precision_at_10'].mean():.3f}",
                f"  mean recall@10         {evaluable['recall_at_10'].mean():.3f}",
                f"  mean recall@20         {evaluable['recall_at_20'].mean():.3f}",
                f"  median event rank      {self.predictions.loc[self.predictions['actual'] == 1, 'rank'].median():.0f}"
                f"  (of ~{int(evaluable['n_scored'].mean())} scored)",
            ]

        if self.skipped:
            lines += ["", f"  origins skipped        {len(self.skipped)}"]
        return "\n".join(lines)


def precision_at_k(ranked_actuals: np.ndarray, k: int) -> float:
    top = ranked_actuals[:k]
    return float(top.sum() / k) if k else float("nan")


def recall_at_k(ranked_actuals: np.ndarray, k: int) -> float:
    total = ranked_actuals.sum()
    return float(ranked_actuals[:k].sum() / total) if total else float("nan")


def walk_forward(
    features: pd.DataFrame,
    *,
    start: tuple[int, int] = (2021, 1),
    end: tuple[int, int] | None = None,
    min_train_events: int = 8,
    feature_names: list[str] | None = None,
    model_factory: Callable[[float], object] = default_model_factory,
) -> BacktestResult:
    """Refit and score quarter by quarter, never using data from after the origin.

    At origin T the training set is every at-risk row whose confirmed label was
    already observable at T -- that is, rows at quarter T-1 or earlier, since
    confirming an onset at quarter q requires seeing q+1. Scoring then happens on
    rows at quarter T, making each prediction a genuine one-quarter-ahead
    forecast.
    """
    feature_names = feature_names or FEATURE_NAMES

    at_risk = features[~features["prev_unaffordable"]].dropna(
        subset=feature_names + [TARGET]
    ).copy()
    at_risk["q"] = quarter_index(at_risk["year"], at_risk["qtr"])

    first = quarter_index(*start)
    last = quarter_index(*end) if end else int(at_risk["q"].max())

    rows: list[dict] = []
    per_origin: list[dict] = []
    skipped: list[dict] = []

    for origin in range(first, last + 1):
        # Labels observable at the origin: an onset at q is confirmed only once
        # q+1 has been seen, so q must be at most origin - 1.
        train = at_risk[at_risk["q"] <= origin - 1]
        score = at_risk[at_risk["q"] == origin]

        year, qtr = divmod(origin, 4)
        label = f"{year}Q{qtr + 1}"

        if score.empty:
            skipped.append({"origin": label, "reason": "nothing to score"})
            continue

        n_train_events = int(train[TARGET].sum())
        if n_train_events < min_train_events:
            skipped.append(
                {"origin": label, "reason": f"only {n_train_events} training events"}
            )
            continue

        X_train, y_train = train[feature_names], train[TARGET].astype(int)
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        model = model_factory(neg / pos if pos else 1.0)
        model.fit(X_train, y_train)

        scores = model.predict_proba(score[feature_names])[:, 1]
        actual = score[TARGET].astype(int).to_numpy()

        order = np.argsort(-scores)
        ranked_actual = actual[order]
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)

        for cbsa, name, s, a, r in zip(
            score["cbsa"], score.get("metro_name", score["cbsa"]), scores, actual, ranks
        ):
            rows.append(
                {
                    "origin": label,
                    "origin_q": origin,
                    "cbsa": cbsa,
                    "metro_name": name,
                    "score": float(s),
                    "actual": int(a),
                    "rank": int(r),
                }
            )

        n_events = int(actual.sum())
        per_origin.append(
            {
                "origin": label,
                "origin_q": origin,
                "n_train": len(train),
                "n_train_events": n_train_events,
                "n_scored": len(score),
                "n_events": n_events,
                "pr_auc": float(average_precision_score(actual, scores))
                if n_events else np.nan,
                "roc_auc": float(roc_auc_score(actual, scores))
                if 0 < n_events < len(actual) else np.nan,
                "precision_at_10": precision_at_k(ranked_actual, 10) if n_events else np.nan,
                "recall_at_10": recall_at_k(ranked_actual, 10) if n_events else np.nan,
                "recall_at_20": recall_at_k(ranked_actual, 20) if n_events else np.nan,
            }
        )

    return BacktestResult(
        predictions=pd.DataFrame(rows),
        per_origin=pd.DataFrame(per_origin),
        skipped=skipped,
    )


def lead_time(result: BacktestResult, *, horizon: int = 8) -> pd.DataFrame:
    """How highly was each metro ranked in the quarters before it collapsed?

    This tests the early-warning claim directly. For every metro that actually
    had a confirmed onset during the backtest, the ranks it held in the preceding
    `horizon` quarters are collected, expressed as a percentile so origins with
    different numbers of scored metros stay comparable.
    """
    predictions = result.predictions
    if predictions.empty:
        return pd.DataFrame()

    scored_per_origin = predictions.groupby("origin_q")["cbsa"].nunique()
    events = predictions[predictions["actual"] == 1][["cbsa", "metro_name", "origin_q"]]

    rows = []
    for _, event in events.iterrows():
        for lag in range(0, horizon + 1):
            prior_q = event["origin_q"] - lag
            match = predictions[
                (predictions["cbsa"] == event["cbsa"])
                & (predictions["origin_q"] == prior_q)
            ]
            if match.empty:
                continue
            rank = int(match["rank"].iloc[0])
            total = int(scored_per_origin.get(prior_q, np.nan))
            rows.append(
                {
                    "cbsa": event["cbsa"],
                    "metro_name": event["metro_name"],
                    "quarters_before_onset": lag,
                    "rank": rank,
                    "n_scored": total,
                    "percentile": 1 - (rank - 1) / total if total else np.nan,
                    "in_top_10": rank <= 10,
                    "in_top_20": rank <= 20,
                }
            )

    return pd.DataFrame(rows)


def lead_time_summary(lead: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lead-time behavior by how far ahead of the onset we are."""
    if lead.empty:
        return pd.DataFrame()
    return (
        lead.groupby("quarters_before_onset")
        .agg(
            events=("cbsa", "size"),
            median_rank=("rank", "median"),
            median_percentile=("percentile", "median"),
            pct_in_top_10=("in_top_10", "mean"),
            pct_in_top_20=("in_top_20", "mean"),
        )
        .reset_index()
    )
