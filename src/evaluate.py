"""Ranking evaluation for the capstone (Step 4).

Implements the CI-aware protocol from the evaluation plan: leave-last-order-out
splitting for repeat buyers, HitRate/Recall@K + MRR + NDCG@10, catalogue
coverage and long-tail exposure, per-method reachability ceilings, and
user-resampled bootstrap confidence intervals. Winners are only claimed when
intervals separate - hit rates over a 33k-product catalogue are tiny numbers.
"""

import numpy as np
import pandas as pd

SEED = 42
K_LEVELS = (10, 50, 100)


def leave_last_order_split(delivered: pd.DataFrame, items: pd.DataFrame,
                           start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split usable-range delivered purchases into (training pairs, held-out
    last-order pairs) for customers with >= 2 orders in [start, end)."""
    o = delivered[
        (delivered["order_purchase_timestamp"] >= start)
        & (delivered["order_purchase_timestamp"] < end)
    ][["order_id", "customer_unique_id", "order_purchase_timestamp"]]
    n_orders = o.groupby("customer_unique_id")["order_id"].nunique()
    repeat = n_orders[n_orders >= 2].index
    last = (o[o["customer_unique_id"].isin(repeat)]
            .sort_values("order_purchase_timestamp")
            .drop_duplicates("customer_unique_id", keep="last"))
    held_ids = set(last["order_id"])
    # training keeps EVERY customer's remaining purchases - one-shot customers'
    # interactions are what give popularity/CF/SVD their signal
    lines = items.merge(o, on="order_id")[
        ["customer_unique_id", "product_id", "order_id"]
    ].drop_duplicates()
    held = lines[lines["order_id"].isin(held_ids)].drop(columns="order_id")
    train = lines[~lines["order_id"].isin(held_ids)].drop(columns="order_id")
    # a held-out product the user ALSO bought earlier stays a valid target
    # (repurchase recommendations are allowed by the seen-item policy)
    return train.drop_duplicates(), held.drop_duplicates()


def rank_metrics(top_items: np.ndarray, held_items: set, k_levels=K_LEVELS,
                 ndcg_k: int = 10) -> dict:
    """Per-user metrics from a best-first ranked item array."""
    hits_at = np.isin(top_items, list(held_items))
    out = {}
    for k in k_levels:
        out[f"hit@{k}"] = float(hits_at[:k].any())
        out[f"recall@{k}"] = float(hits_at[:k].sum() / len(held_items))
    ranks = np.nonzero(hits_at)[0]
    out["mrr"] = float(1.0 / (ranks[0] + 1)) if len(ranks) else 0.0
    dcg = float((hits_at[:ndcg_k] / np.log2(np.arange(2, ndcg_k + 2))).sum())
    ideal = min(len(held_items), ndcg_k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal + 2))).sum())
    out[f"ndcg@{ndcg_k}"] = dcg / idcg if idcg > 0 else 0.0
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the mean, resampling users."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = np.asarray(values)[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def coverage_and_longtail(all_top10: list[np.ndarray], popularity: np.ndarray,
                          n_items: int) -> dict:
    """Catalogue coverage of top-10 slots + share of slots outside the top
    popularity decile (provider-side exposure)."""
    stacked = np.concatenate(all_top10)
    top_decile_cut = np.quantile(popularity[popularity > 0], 0.9) if (popularity > 0).any() else 0
    head_items = set(np.nonzero(popularity >= max(top_decile_cut, 1))[0])
    longtail = np.mean([i not in head_items for i in stacked])
    return {
        "coverage": len(np.unique(stacked)) / n_items,
        "longtail_share": float(longtail),
    }


def reachability(held: pd.DataFrame, train_pairs: pd.DataFrame) -> float:
    """Share of held-out products that appear in the training interactions -
    the ceiling for CF/SVD, which cannot recommend unseen items."""
    seen = set(train_pairs["product_id"])
    return float(held["product_id"].isin(seen).mean())


def evaluate_model(name: str, score_fn, im, eval_users: pd.DataFrame,
                   k_max: int = 100) -> pd.DataFrame:
    """Run one candidate generator over all eval users.

    score_fn(customer_id, user_items) -> full score vector. eval_users maps
    customer_unique_id -> set of held-out item indices.
    """
    from src.models.candidate_gen import top_k_from_scores

    rows = []
    top10s = []
    for cust, held_idx in eval_users.items():
        user_items = im.user_items(cust)
        scores = score_fn(cust, user_items)
        top = top_k_from_scores(scores, k_max)
        m = rank_metrics(top, held_idx)
        m["customer_unique_id"] = cust
        m["repurchase_hit"] = float(bool(set(top[:10]) & held_idx & set(user_items)))
        rows.append(m)
        top10s.append(top[:10])
    df = pd.DataFrame(rows)
    df.attrs["top10s"] = top10s
    df.attrs["model"] = name
    return df
