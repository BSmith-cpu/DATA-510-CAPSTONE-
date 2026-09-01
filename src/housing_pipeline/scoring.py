"""Turning raw model output into something a stakeholder can act on.

The model is trained on a deliberately enriched population -- only metros that
had a confirmed collapse, plus near-miss metros -- because the natural rate of
confirmed onsets among at-risk metros is roughly 1%, too sparse to learn from.
That enrichment is the right modeling choice and the wrong scoring assumption:
the probabilities it produces are calibrated to a population about four times
denser in events than the one being scored.

Left uncorrected this is not a subtle bias. The tuned operating threshold sat at
0.31 while the highest-scoring metro in the deployment population scored 0.16,
so the alert could never fire at all.

Two outputs are produced here, and the distinction matters:

``risk_rank`` / ``risk_percentile`` / ``risk_tier``
    Relative standing within the scored population. Always valid, because rank
    ordering is exactly what the grouped cross-validation measured. This is the
    primary product.

``risk_probability``
    The raw score shifted from the training prior onto the deployment prior.
    Interpretable as a probability, but only as good as the base-rate estimate
    it is anchored to -- which is why that estimate is computed from observed
    data rather than assumed, and is reported alongside the scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Percentile cutoffs for the tier labels. These describe standing within the
# scored population; they are not thresholds on an absolute probability.
TIER_ELEVATED = 0.95
TIER_WATCH = 0.80

DEFAULT_TARGET = "collapse_onset_confirmed"


@dataclass(frozen=True)
class BaseRates:
    """Positive rates in the population trained on versus the one scored."""

    training: float
    deployment: float

    @property
    def enrichment(self) -> float:
        """How many times denser in events the training population is."""
        return self.training / self.deployment if self.deployment else float("nan")

    def describe(self) -> str:
        return (
            f"training {self.training:.3%} vs deployment {self.deployment:.3%} "
            f"({self.enrichment:.1f}x enriched)"
        )


def deployment_base_rate(
    at_risk: pd.DataFrame, target: str = DEFAULT_TARGET
) -> float:
    """Observed rate of confirmed onsets across the whole at-risk population.

    This is the honest anchor for prior correction: it is measured over every
    at-risk metro-quarter with complete features, not just the ones the model
    was trained on.
    """
    if at_risk.empty:
        raise ValueError("cannot estimate a base rate from an empty population")
    return float(at_risk[target].mean())


def base_rates(
    training_pool: pd.DataFrame,
    at_risk: pd.DataFrame,
    target: str = DEFAULT_TARGET,
) -> BaseRates:
    return BaseRates(
        training=float(training_pool[target].mean()),
        deployment=deployment_base_rate(at_risk, target=target),
    )


def prior_correct(
    scores: np.ndarray | pd.Series,
    rates: BaseRates,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Shift probabilities from the training prior onto the deployment prior.

    Standard prior-shift (case-control) correction. It assumes the likelihood
    ratio the model learned transfers even though the class balance does not,
    which is the usual assumption behind training on an enriched sample:

        p' = p·r / (p·r + (1-p)·s)

    with ``r = π_deploy / π_train`` and ``s = (1-π_deploy) / (1-π_train)``.

    Because the deployment prior is lower than the training prior, every
    corrected probability is lower than its input -- the ordering is unchanged,
    which is why the ranking is unaffected by whether this is applied.
    """
    p = np.clip(np.asarray(scores, dtype=float), eps, 1 - eps)

    r = rates.deployment / max(rates.training, eps)
    s = (1 - rates.deployment) / max(1 - rates.training, eps)

    corrected = (p * r) / (p * r + (1 - p) * s)
    return np.clip(corrected, 0.0, 1.0)


def assign_tiers(percentile: pd.Series) -> pd.Series:
    """Label relative standing. Percentile-based, deliberately not probability-based."""
    return pd.Series(
        np.select(
            [percentile >= TIER_ELEVATED, percentile >= TIER_WATCH],
            ["Elevated", "Watch"],
            default="Monitor",
        ),
        index=percentile.index,
        dtype="object",
    )


def build_watchlist(
    holdout: pd.DataFrame,
    raw_scores: np.ndarray | pd.Series,
    rates: BaseRates | None = None,
    *,
    metro_col: str = "metro_name",
) -> pd.DataFrame:
    """Aggregate row-level model scores into a ranked per-metro watchlist.

    Parameters
    ----------
    holdout:
        Scored rows, one per metro-quarter, carrying ``cbsa`` and a metro name.
    raw_scores:
        Model output for those rows, on the training-population scale.
    rates:
        Training and deployment base rates. When supplied, a prior-corrected
        ``risk_probability`` column is added; when omitted, only the ranking is
        produced -- which is the honest default if the deployment rate cannot be
        estimated.

    Returns
    -------
    One row per metro, ranked, with ``risk_rank`` 1 as the highest risk.
    """
    scored = holdout[["cbsa", metro_col]].copy()
    scored["risk_score_raw"] = np.asarray(raw_scores, dtype=float)

    # A metro contributes many quarters; its standing is the average of them.
    per_metro = (
        scored.groupby(["cbsa", metro_col], as_index=False)["risk_score_raw"]
        .mean()
        .sort_values("risk_score_raw", ascending=False)
        .reset_index(drop=True)
    )

    per_metro["risk_rank"] = np.arange(1, len(per_metro) + 1)
    per_metro["risk_percentile"] = per_metro["risk_score_raw"].rank(pct=True)
    per_metro["risk_tier"] = assign_tiers(per_metro["risk_percentile"])

    if rates is not None:
        per_metro["risk_probability"] = prior_correct(
            per_metro["risk_score_raw"], rates
        )
        per_metro.attrs["base_rates"] = rates

    return per_metro


def watchlist_summary(watchlist: pd.DataFrame, top_n: int = 15) -> str:
    """A short, printable summary of the watchlist and how to read it."""
    lines = [
        f"Watchlist: {len(watchlist)} metros ranked by early-warning risk.",
        "",
        "Read this as relative standing, not an absolute probability of collapse.",
    ]

    rates = watchlist.attrs.get("base_rates")
    if rates is not None:
        lines.append(
            f"Base rates: {rates.describe()}; risk_probability is prior-corrected "
            f"onto the deployment rate."
        )

    counts = watchlist["risk_tier"].value_counts()
    lines.append("")
    for tier in ("Elevated", "Watch", "Monitor"):
        if tier in counts:
            lines.append(f"  {tier:<9} {counts[tier]:>4} metros")

    lines.append("")
    lines.append(f"Top {top_n}:")
    columns = ["risk_rank", "metro_name", "risk_tier", "risk_score_raw"]
    if "risk_probability" in watchlist.columns:
        columns.append("risk_probability")
    available = [c for c in columns if c in watchlist.columns]
    lines.append(watchlist.head(top_n)[available].to_string(index=False))

    return "\n".join(lines)
