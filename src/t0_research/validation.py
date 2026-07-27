"""Date-blocked validation with purging and embargo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def clustered_mean_lower_bound(
    values: pd.Series,
    clusters: pd.Series,
    *,
    z_value: float = 1.2815515655446004,
) -> tuple[float, float, int]:
    """Return mean, cluster-robust SE proxy and a one-sided lower bound.

    Observations are averaged within cluster first, so many signals from one
    trading day do not masquerade as independent evidence.
    """
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "cluster": clusters})
    grouped = frame.dropna().groupby("cluster")["value"].mean()
    n_clusters = len(grouped)
    mean = float(grouped.mean()) if n_clusters else np.nan
    if n_clusters <= 1:
        return mean, np.nan, n_clusters
    se = float(grouped.std(ddof=1) / np.sqrt(n_clusters))
    return mean, mean - z_value * se, n_clusters


@dataclass(frozen=True)
class PurgedDateSplit:
    """Expanding date folds for overlapping forward-return labels."""

    min_train_days: int = 60
    test_days: int = 20
    purge_days: int = 1
    embargo_days: int = 1

    def split(self, dates: pd.Series | np.ndarray):
        values = pd.Series(dates).astype(str)
        unique = np.array(sorted(values.dropna().unique()))
        start = self.min_train_days + self.purge_days
        while start + self.test_days <= len(unique):
            test_dates = unique[start : start + self.test_days]
            train_end = start - self.purge_days
            train_dates = unique[:train_end]
            train_idx = np.flatnonzero(values.isin(train_dates).to_numpy())
            test_idx = np.flatnonzero(values.isin(test_dates).to_numpy())
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx
            start += self.test_days + self.embargo_days
