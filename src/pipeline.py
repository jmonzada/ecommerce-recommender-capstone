"""Shared serving-pipeline assembly (used by notebook 04 and the app).

Rebuilds the feature-window artefacts, candidate lists, and ranker feature
matrices exactly as notebook 03 trained them (same seeds, same windows, same
config), so the audit and the API score the same pipeline the evaluation
chose.
"""

import numpy as np
import pandas as pd
import yaml

from src.data import REPO_ROOT, load_table
from src.features import (
    cold_fill_stats,
    customer_features,
    customer_geo,
    interactions,
    load_centroids,
    load_clean_items,
    load_clean_orders,
    load_clean_products,
    load_clean_reviews,
    load_windows,
    pair_features,
    product_features,
    seller_features,
)
from src.models.candidate_gen import (
    ContentBased,
    HybridRecommender,
    InteractionMatrix,
    ItemItemCF,
    PopularityRecommender,
    top_k_from_scores,
)
from src.recommend import Router, regional_popularity


def build_artifacts() -> dict:
    """Feature-window artefacts + feature tables, per models/chosen_config.yaml."""
    chosen = yaml.safe_load((REPO_ROOT / "models" / "chosen_config.yaml").read_text())
    windows = load_windows()
    fw = windows["feature_window"]

    delivered, _ = load_clean_orders()
    items, _ = load_clean_items()
    reviews, _ = load_clean_reviews()
    products, _ = load_clean_products()
    payments = load_table("order_payments")
    centroids = load_centroids()
    geo = customer_geo()

    fw_inter = interactions(delivered, items, *fw)
    im_fw = InteractionMatrix(
        fw_inter[["customer_unique_id", "product_id"]].drop_duplicates(),
        products["product_id"].to_numpy(),
    )
    cf_fw = ItemItemCF().fit(im_fw)
    content_fw = ContentBased().fit(im_fw, products.set_index("product_id"))
    hybrid = HybridRecommender(cf_fw, content_fw, w=chosen["hybrid_w"])
    pop_fw = PopularityRecommender().fit(im_fw)
    n_cand = chosen["n_candidates"]
    router = Router(
        im_fw, hybrid,
        global_top=top_k_from_scores(pop_fw.scores_, n_cand),
        regional_top=regional_popularity(fw_inter, im_fw.item_index, k=n_cand),
        n_candidates=n_cand,
    )
    return {
        "windows": windows, "chosen": chosen,
        "delivered": delivered, "items": items, "reviews": reviews,
        "products": products, "payments": payments, "centroids": centroids,
        "geo": geo, "geo_region": geo["region"].to_dict(),
        "fw_inter": fw_inter, "im_fw": im_fw, "cf_fw": cf_fw,
        "content_fw": content_fw, "hybrid": hybrid, "pop_fw": pop_fw,
        "router": router,
        "cust_fw": customer_features(delivered, items, payments, reviews, products, fw),
        "prod_fw": product_features(delivered, items, reviews, products, fw),
        "sell_fw": seller_features(delivered, items, reviews, fw),
        "fills": cold_fill_stats(delivered, items, reviews, centroids, fw),
    }


def candidate_frame(art: dict, customers) -> tuple[pd.DataFrame, pd.Series]:
    """Stage-1 candidate lists (with ranks) + route labels for `customers`."""
    rows, route_of = [], {}
    router, im_fw = art["router"], art["im_fw"]
    for cust in customers:
        cands, route = router.recommend(cust, region=art["geo_region"].get(cust), k=None)
        route_of[cust] = route
        rows.append(pd.DataFrame({
            "customer_unique_id": cust,
            "product_id": im_fw.product_ids[cands],
            "stage1_rank": np.arange(len(cands)),
        }))
    return pd.concat(rows, ignore_index=True), pd.Series(route_of)


def ranker_matrix(pairs: pd.DataFrame, art: dict,
                  columns: list[str] | None = None) -> pd.DataFrame:
    """Ranker feature matrix as trained in notebook 03 (incl. cf_signal and
    encodings). Pass `columns` to align to the training layout."""
    x_all = pair_features(pairs[["customer_unique_id", "product_id"]],
                          art["cust_fw"], art["geo"], art["prod_fw"],
                          art["sell_fw"], art["centroids"], art["fills"])
    user_items_map = {c: art["im_fw"].user_items(c)
                      for c in pairs["customer_unique_id"].unique()}
    x_all["cf_signal"] = art["cf_fw"].pair_scores(user_items_map, pairs,
                                                  art["im_fw"].item_index)
    x_all["has_cf_signal"] = (x_all["cf_signal"] > 0).astype(float)
    cat_share = (art["fw_inter"]
                 .merge(art["products"][["product_id", "category"]], on="product_id")
                 ["category"].value_counts(normalize=True))
    num = x_all.drop(columns=["customer_unique_id", "product_id", "p_category", "c_region"])
    num["p_category_freq"] = x_all["p_category"].map(cat_share).fillna(0.0)
    num = pd.concat(
        [num, pd.get_dummies(x_all["c_region"], prefix="region", dtype=float)], axis=1
    )
    if columns is not None:
        num = num.reindex(columns=columns, fill_value=0.0)
    return num


def recommend_two_stage(art: dict, ranker, feature_columns: list[str],
                        customer_id: str, k: int = 10) -> tuple[pd.DataFrame, str]:
    """The final Step 4 pipeline for one customer: routed Stage-1 candidates,
    re-ranked by the trained ranker. Returns (top-k frame with scores and
    product context, route label)."""
    region = art["geo_region"].get(customer_id)
    cands, route = art["router"].recommend(customer_id, region=region, k=None)
    pairs = pd.DataFrame({
        "customer_unique_id": customer_id,
        "product_id": art["im_fw"].product_ids[cands],
    })
    x = ranker_matrix(pairs, art, columns=feature_columns)
    pairs["score"] = ranker.predict_proba(x)[:, 1]
    if route == "hybrid_only":
        route = "two_stage"  # the re-rank below is the second stage
    top = pairs.sort_values("score", ascending=False).head(k).copy()
    prod = art["prod_fw"]
    top["category"] = top["product_id"].map(prod["category"])
    top["price_brl"] = top["product_id"].map(prod["median_price_w"]).round(2)
    top["popularity"] = top["product_id"].map(prod["popularity"]).astype(int)
    return top.reset_index(drop=True), route
