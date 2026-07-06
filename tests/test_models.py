"""Tests for candidate generators, ranking metrics, and routing on toy data."""

import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, rank_metrics
from src.models.candidate_gen import (
    HybridRecommender,
    InteractionMatrix,
    ItemItemCF,
    PopularityRecommender,
    top_k_from_scores,
)
from src.recommend import Router

PRODUCTS = np.array(["a", "b", "c", "d"])

# u1 buys a+b, u2 buys a+b, u3 buys c -> a and b are perfect co-purchase partners
PAIRS = pd.DataFrame({
    "customer_unique_id": ["u1", "u1", "u2", "u2", "u3"],
    "product_id": ["a", "b", "a", "b", "c"],
})


def test_interaction_matrix_shape_and_lookup():
    im = InteractionMatrix(PAIRS, PRODUCTS)
    assert im.X.shape == (3, 4)
    assert set(im.user_items("u1")) == {0, 1}
    assert len(im.user_items("stranger")) == 0


def test_item_item_cf_scores_copurchase_partner_highest():
    im = InteractionMatrix(PAIRS, PRODUCTS)
    cf = ItemItemCF().fit(im)
    # user who bought only "a" -> "b" must outscore "c" and "d"
    scores = cf.scores(im, np.array([0]))
    assert scores[1] > scores[2] and scores[1] > scores[3]
    assert scores[0] == 0  # self-similarity zeroed

    pair_scores = cf.pair_scores(
        {"u3": np.array([2])},
        pd.DataFrame({"customer_unique_id": ["u3", "u3"], "product_id": ["a", "d"]}),
        im.item_index,
    )
    assert pair_scores[0] == 0 and pair_scores[1] == 0  # c has no co-purchases


def test_popularity_and_topk():
    im = InteractionMatrix(PAIRS, PRODUCTS)
    pop = PopularityRecommender().fit(im)
    top = top_k_from_scores(pop.scores(im, np.array([])), 2)
    assert set(top) == {0, 1}  # a and b each bought twice


def test_hybrid_blend_bounds():
    im = InteractionMatrix(PAIRS, PRODUCTS)
    cf = ItemItemCF().fit(im)

    class FakeContent:
        def scores(self, im, user_items):
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    hy = HybridRecommender(cf, FakeContent(), w=0.0)  # pure content
    assert top_k_from_scores(hy.scores(im, np.array([0])), 1)[0] == 2


def test_rank_metrics_known_values():
    m = rank_metrics(np.array([5, 3, 9, 1]), held_items={3}, k_levels=(1, 2), ndcg_k=2)
    assert m["hit@1"] == 0.0 and m["hit@2"] == 1.0
    assert m["mrr"] == 0.5
    assert abs(m["ndcg@2"] - (1 / np.log2(3))) < 1e-9


def test_bootstrap_ci_contains_mean():
    vals = np.array([0.0, 1.0] * 50)
    lo, hi = bootstrap_ci(vals, n_boot=200)
    assert lo <= 0.5 <= hi


def test_router_routes():
    im = InteractionMatrix(PAIRS, PRODUCTS)
    cf = ItemItemCF().fit(im)

    class FakeContent:
        def scores(self, im, user_items):
            return np.zeros(4, dtype=np.float32)

    router = Router(
        im, HybridRecommender(cf, FakeContent(), w=1.0),
        global_top=np.array([0, 1, 2]),
        regional_top={"South": np.array([3, 2])},
    )
    _, route = router.recommend("u1")
    assert route == "hybrid_only"
    items, route = router.recommend("stranger", region="South")
    assert route == "regional_popularity" and items[0] == 3
    _, route = router.recommend("stranger", region="Mars")
    assert route == "global_popularity"
