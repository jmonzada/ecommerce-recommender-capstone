"""Fairness audit utilities (capstone Step 5).

Implements the pinned protocol: per-group metrics carry n and bootstrap CIs,
groups below MIN_GROUP_N are suppressed rather than reported as noise, group
AUC/TPR require a minimum number of positives, demographic parity and
disparate impact are computed for descriptive reporting (the primary criteria
are separation-style: per-group TPR and AUC), and the exposure side gets
Gini/Lorenz, long-tail share, and an MMR-style popularity-penalty re-ranker.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import bootstrap_ci

MIN_GROUP_N = 30      # groups smaller than this get suppressed, not reported
MIN_POS_FOR_RATE = 20  # TPR/AUC need at least this many positives to mean anything


def group_metric_table(y, scores, groups, tau) -> pd.DataFrame:
    """Per-group audit table on a shared evaluation population.

    Columns: n, n_pos, base_rate, selection_rate (share of pairs with
    score >= tau), tpr (share of true positives with score >= tau, with
    bootstrap CI), auc. Metrics for groups with n < MIN_GROUP_N are NaN by
    design; TPR/AUC additionally require MIN_POS_FOR_RATE positives.
    """
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores),
                       "g": np.asarray(groups)})
    rows = []
    for g, grp in df.groupby("g"):
        row = {"group": g, "n": len(grp), "n_pos": int(grp["y"].sum()),
               "base_rate": grp["y"].mean()}
        if len(grp) >= MIN_GROUP_N:
            row["selection_rate"] = float((grp["s"] >= tau).mean())
            pos = grp[grp["y"] == 1]
            if len(pos) >= MIN_POS_FOR_RATE:
                hits = (pos["s"] >= tau).to_numpy().astype(float)
                row["tpr"] = float(hits.mean())
                row["tpr_lo"], row["tpr_hi"] = bootstrap_ci(hits)
                if row["n_pos"] < len(grp):
                    row["auc"] = roc_auc_score(grp["y"], grp["s"])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("group")
    sel = table["selection_rate"].dropna()
    table.attrs["dp_difference"] = float(sel.max() - sel.min()) if len(sel) > 1 else np.nan
    table.attrs["di_ratio"] = float(sel.min() / sel.max()) if len(sel) > 1 and sel.max() > 0 else np.nan
    tpr = table["tpr"].dropna() if "tpr" in table else pd.Series(dtype=float)
    table.attrs["tpr_gap"] = float(tpr.max() - tpr.min()) if len(tpr) > 1 else np.nan
    return table


def exposure_gini(exposure_counts: np.ndarray) -> float:
    """Gini coefficient of recommendation exposure across the catalogue
    (zeros included - unexposed products are part of the inequality)."""
    x = np.sort(np.asarray(exposure_counts, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def lorenz_points(exposure_counts: np.ndarray, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """(population share, exposure share) points for a Lorenz curve."""
    x = np.sort(np.asarray(exposure_counts, dtype=float))
    cum = np.cumsum(x) / max(x.sum(), 1)
    idx = np.linspace(0, len(x) - 1, n_points).astype(int)
    return (idx + 1) / len(x), cum[idx]


def rerank_with_popularity_penalty(scored: pd.DataFrame, lam: float,
                                   score_col: str = "p",
                                   pop_col: str = "pop_pct",
                                   k: int = 10) -> dict:
    """MMR-style exposure mitigation: score' = score - lam * popularity
    percentile. Returns {customer -> top-k product_id array}. lam=0 reproduces
    the unmitigated ranking."""
    s = scored.copy()
    s["_adj"] = s[score_col] - lam * s[pop_col]
    return {c: g.sort_values("_adj", ascending=False)["product_id"].to_numpy()[:k]
            for c, g in s.groupby("customer_unique_id")}
