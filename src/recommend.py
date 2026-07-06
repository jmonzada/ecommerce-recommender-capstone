"""Serving-time recommendation routing (capstone Step 4, core scope).

Routing rule:
- customer with feature-window purchase history -> hybrid candidates, optionally
  re-ranked by the trained ranker (the two-stage pipeline)
- cold customer with known region -> regional popularity
- unknown customer -> global popularity

The router reports which route served each request so routing shares can be
measured (they are part of the Step 1 KPI scoping).
"""

import numpy as np
import pandas as pd

from src.models.candidate_gen import top_k_from_scores


def regional_popularity(fw_interactions: pd.DataFrame, item_index: pd.Index,
                        k: int = 100) -> dict[str, np.ndarray]:
    """Top-k item indices per customer region, from feature-window purchases."""
    out = {}
    for region, grp in fw_interactions.groupby("region"):
        counts = grp["product_id"].value_counts()
        idx = item_index.get_indexer(counts.index)
        out[region] = idx[idx >= 0][:k]
    return out


class Router:
    """Composes the serving decision. `rerank_fn(customer_id, candidate_idx)`
    returns candidate indices re-ordered by the ranker; None serves raw hybrid."""

    def __init__(self, im, hybrid, global_top: np.ndarray,
                 regional_top: dict[str, np.ndarray], rerank_fn=None,
                 n_candidates: int = 50):
        self.im = im
        self.hybrid = hybrid
        self.global_top = global_top
        self.regional_top = regional_top
        self.rerank_fn = rerank_fn
        self.n_candidates = n_candidates

    def recommend(self, customer_id, region: str | None = None,
                  k: int = 10) -> tuple[np.ndarray, str]:
        """Returns (item indices best-first, route label)."""
        user_items = self.im.user_items(customer_id)
        if len(user_items) > 0:
            scores = self.hybrid.scores(self.im, user_items)
            cands = top_k_from_scores(scores, self.n_candidates)
            if self.rerank_fn is not None:
                return self.rerank_fn(customer_id, cands)[:k], "two_stage"
            return cands[:k], "hybrid_only"
        if region is not None and region in self.regional_top:
            return self.regional_top[region][:k], "regional_popularity"
        return self.global_top[:k], "global_popularity"
