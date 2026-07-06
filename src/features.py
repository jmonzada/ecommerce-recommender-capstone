"""Feature engineering for the two-stage recommender (capstone Step 3).

Every aggregate here obeys the three-window scheme in configs/windows.yaml:
functions take explicit window bounds and filter their inputs before
aggregating, so training and holdout pairs alike are scored only with
information available at the feature-window cutoff. `assert_max_ts` is the
audit hook the notebooks use to prove no input postdates its cutoff.

Cleaning rules (each returns per-rule drop/impute counts — no silent cleaning):
- orders: keep `delivered` only; parse timestamps; attach customer_unique_id
- reviews: keep the latest review per order (review_id is not unique)
- order_items: collapse unit rows to one line per (order, product, seller)
- products: category -> English (fallback 'unknown'), median-impute attributes
"""

import numpy as np
import pandas as pd
import yaml

from src.data import BR_STATE_REGION, GEO_CENTROIDS_PATH, REPO_ROOT, load_table

WINDOWS_PATH = REPO_ROOT / "configs" / "windows.yaml"
PRICE_WINSOR_Q = 0.99  # heavy right tails (p99 ~ R$890 vs max R$6,735)

PRODUCT_ATTR_COLS = [
    "product_name_lenght", "product_description_lenght", "product_photos_qty",
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
]


def load_windows() -> dict:
    """Half-open [start, end) window bounds from configs/windows.yaml."""
    raw = yaml.safe_load(WINDOWS_PATH.read_text())
    return {
        key: (pd.Timestamp(raw[key]["start"]), pd.Timestamp(raw[key]["end"]) + pd.Timedelta(days=1))
        for key in ("feature_window", "label_window", "holdout")
    }


def assert_max_ts(df: pd.DataFrame, col: str, cutoff: pd.Timestamp) -> None:
    """Leakage audit: fail loudly if any timestamp in `col` is >= cutoff."""
    mx = df[col].max()
    if pd.notna(mx) and mx >= cutoff:
        raise AssertionError(f"leakage: max {col} = {mx} >= cutoff {cutoff}")


def haversine_km(lat1, lng1, lat2, lng2):
    """Vectorised great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lng2) - np.asarray(lng1)) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- cleaning

def load_clean_orders() -> tuple[pd.DataFrame, dict]:
    orders = load_table("orders")
    counts = {"orders_raw": len(orders)}
    ts_cols = [c for c in orders.columns if c.endswith(("_date", "_timestamp"))]
    orders[ts_cols] = orders[ts_cols].apply(pd.to_datetime)
    delivered = orders[orders["order_status"] == "delivered"].copy()
    counts["dropped_not_delivered"] = counts["orders_raw"] - len(delivered)
    cust = load_table("customers")[
        ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_state"]
    ]
    delivered = delivered.merge(cust, on="customer_id", how="left")
    delivered["region"] = delivered["customer_state"].map(BR_STATE_REGION)
    return delivered, counts


def load_clean_reviews() -> tuple[pd.DataFrame, dict]:
    reviews = load_table("order_reviews")
    counts = {"reviews_raw": len(reviews)}
    for c in ("review_creation_date", "review_answer_timestamp"):
        reviews[c] = pd.to_datetime(reviews[c])
    reviews = reviews.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    counts["dropped_duplicate_reviews"] = counts["reviews_raw"] - len(reviews)
    return reviews, counts


def reviews_before(reviews: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Reviews created strictly before `cutoff`.

    A feature-window ORDER can carry a review written weeks later (6.7% of
    reviews on feature-window orders were created after the window cutoff), so
    review-based aggregates must filter on review_creation_date, not just join
    by order_id. Creation date is the conservative bound - the survey exists
    then even if the customer answers later."""
    return reviews[reviews["review_creation_date"] < cutoff]


def load_clean_items() -> tuple[pd.DataFrame, dict]:
    items = load_table("order_items")
    counts = {"item_rows_raw": len(items)}
    items = items.groupby(["order_id", "product_id", "seller_id"], as_index=False).agg(
        quantity=("order_item_id", "size"),
        price=("price", "sum"),          # line total
        freight_value=("freight_value", "sum"),
        unit_price=("price", "mean"),
    )
    counts["item_lines_after_unit_dedupe"] = len(items)
    return items, counts


def load_clean_products() -> tuple[pd.DataFrame, dict]:
    products = load_table("products")
    counts = {"missing_category": int(products["product_category_name"].isna().sum())}
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    translation = load_table("category_translation")
    products = products.merge(translation, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )
    counts["imputed_attribute_values"] = int(products[PRODUCT_ATTR_COLS].isna().sum().sum())
    products[PRODUCT_ATTR_COLS] = products[PRODUCT_ATTR_COLS].fillna(
        products[PRODUCT_ATTR_COLS].median()
    )
    return products.drop(columns=["product_category_name", "product_category_name_english"]), counts


def interactions(delivered: pd.DataFrame, items: pd.DataFrame, start, end) -> pd.DataFrame:
    """Item lines for delivered orders purchased in [start, end)."""
    o = delivered[
        (delivered["order_purchase_timestamp"] >= start)
        & (delivered["order_purchase_timestamp"] < end)
    ]
    keep = [
        "order_id", "customer_unique_id", "order_purchase_timestamp",
        "order_delivered_customer_date", "customer_state",
        "customer_zip_code_prefix", "region",
    ]
    return items.merge(o[keep], on="order_id")


# ---------------------------------------------------------------- features

def customer_features(delivered, items, payments, reviews, products, window) -> pd.DataFrame:
    """Per customer_unique_id, from feature-window purchases only: RFM, spend
    profile, review behaviour, preferred category, geography."""
    start, end = window
    inter = interactions(delivered, items, start, end)
    inter = inter.merge(products[["product_id", "category"]], on="product_id", how="left")

    order_level = inter.groupby(["customer_unique_id", "order_id"]).agg(
        order_value=("price", "sum"), ts=("order_purchase_timestamp", "first")
    ).reset_index()
    feats = order_level.groupby("customer_unique_id").agg(
        frequency=("order_id", "size"),
        monetary=("order_value", "sum"),
        last_ts=("ts", "max"),
    )
    feats["recency_days"] = (end - feats.pop("last_ts")).dt.days
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    feats["median_item_price"] = inter.groupby("customer_unique_id")["unit_price"].median()

    pay = payments.merge(inter[["order_id"]].drop_duplicates(), on="order_id")
    feats["avg_installments"] = pay.groupby("order_id")["payment_installments"].max().to_frame().join(
        inter[["order_id", "customer_unique_id"]].drop_duplicates().set_index("order_id")
    ).groupby("customer_unique_id")["payment_installments"].mean()

    rev = reviews_before(reviews, end).merge(
        inter[["order_id", "customer_unique_id"]].drop_duplicates(), on="order_id"
    )
    feats["avg_review_given"] = rev.groupby("customer_unique_id")["review_score"].mean()

    feats["preferred_category"] = (
        inter.groupby(["customer_unique_id", "category"]).size().reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("customer_unique_id")
        .set_index("customer_unique_id")["category"]
    )
    geo = inter.sort_values("order_purchase_timestamp").drop_duplicates("customer_unique_id", keep="last")
    feats["state"] = geo.set_index("customer_unique_id")["customer_state"]
    feats["zip_code_prefix"] = geo.set_index("customer_unique_id")["customer_zip_code_prefix"]
    feats["region"] = feats["state"].map(BR_STATE_REGION)
    return feats


def product_features(delivered, items, reviews, products, window) -> pd.DataFrame:
    """Per product, from feature-window sales only, joined to static attributes.
    Cold products (no window sales) get popularity 0 and category-median prices,
    flagged with has_sales=0. 'category_satisfaction_rate' stands in for the
    planned 'category conversion rate' - Olist has no view/impression data, so
    conversion is unobservable; satisfaction (share of reviews >= 4) is the
    closest measurable analogue."""
    start, end = window
    inter = interactions(delivered, items, start, end)
    rev = reviews_before(reviews, end).merge(
        inter[["order_id", "product_id"]].drop_duplicates(), on="order_id"
    )

    sold = inter.groupby("product_id").agg(
        popularity=("order_id", "nunique"),
        median_price=("unit_price", "median"),
        mean_freight=("freight_value", "mean"),
    )
    sold["review_mean"] = rev.groupby("product_id")["review_score"].mean()

    feats = products.set_index("product_id").join(sold)
    feats["has_sales"] = feats["popularity"].notna().astype(int)
    feats["popularity"] = feats["popularity"].fillna(0)

    cat_median_price = feats.groupby("category")["median_price"].transform("median")
    feats["median_price"] = feats["median_price"].fillna(cat_median_price)
    feats["median_price"] = feats["median_price"].fillna(feats["median_price"].median())
    feats["mean_freight"] = feats["mean_freight"].fillna(feats["mean_freight"].median())
    feats["review_mean"] = feats["review_mean"].fillna(rev["review_score"].mean())

    cap = feats["median_price"].quantile(PRICE_WINSOR_Q)
    feats["median_price_w"] = feats["median_price"].clip(upper=cap)
    feats["freight_ratio"] = (feats["mean_freight"] / feats["median_price_w"]).clip(upper=5)
    feats["price_band"] = pd.qcut(feats["median_price_w"], 5, labels=False, duplicates="drop")

    cat_sat = (rev.merge(products[["product_id", "category"]], on="product_id")
               .groupby("category")["review_score"].apply(lambda s: (s >= 4).mean()))
    feats["category_satisfaction_rate"] = feats["category"].map(cat_sat).fillna(cat_sat.mean())

    main_seller = (inter.groupby(["product_id", "seller_id"]).size().reset_index(name="n")
                   .sort_values("n", ascending=False).drop_duplicates("product_id")
                   .set_index("product_id")["seller_id"])
    feats["main_seller_id"] = main_seller
    return feats


def seller_features(delivered, items, reviews, window) -> pd.DataFrame:
    """Per seller, from feature-window sales: volume, review mean, geography."""
    start, end = window
    inter = interactions(delivered, items, start, end)
    rev = reviews_before(reviews, end).merge(
        inter[["order_id", "seller_id"]].drop_duplicates(), on="order_id"
    )
    feats = inter.groupby("seller_id").agg(seller_order_count=("order_id", "nunique"))
    feats["seller_review_mean"] = rev.groupby("seller_id")["review_score"].mean()
    feats["seller_review_mean"] = feats["seller_review_mean"].fillna(rev["review_score"].mean())
    sellers = load_table("sellers").set_index("seller_id")
    feats["seller_state"] = sellers["seller_state"]
    feats["seller_zip_code_prefix"] = sellers["seller_zip_code_prefix"]
    return feats


def load_centroids() -> pd.DataFrame:
    return pd.read_csv(GEO_CENTROIDS_PATH).set_index("zip_code_prefix")


def customer_geo() -> pd.DataFrame:
    """Static geography per customer_unique_id (modal zip/state across their
    order-customer rows). An address is known at serving time, so unlike the
    behavioural aggregates this is NOT window-filtered - cold customers keep
    their geography."""
    cust = load_table("customers")
    modal = (
        cust.groupby(["customer_unique_id", "customer_state", "customer_zip_code_prefix"])
        .size().reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("customer_unique_id")
        .set_index("customer_unique_id")
    )
    modal["region"] = modal["customer_state"].map(BR_STATE_REGION)
    return modal[["customer_state", "customer_zip_code_prefix", "region"]]


def cold_fill_stats(delivered, items, reviews, centroids, window) -> dict:
    """Cold-start fill constants pinned to feature-window statistics.

    Computing fills from the scoring batch itself would make a pair's features
    depend on which batch it is scored with, and would let validation rows
    shape the fills applied to training rows. These are fixed once per window:
    max recency = window length, review fills = window review mean, distance
    fill = median customer->seller distance over actual window purchases."""
    start, end = window
    inter = interactions(delivered, items, start, end)
    rev = reviews_before(reviews, end).merge(inter[["order_id"]].drop_duplicates(), on="order_id")
    sellers = load_table("sellers")[["seller_id", "seller_zip_code_prefix"]]
    d = inter.merge(sellers, on="seller_id")
    dist = haversine_km(
        d["customer_zip_code_prefix"].map(centroids["lat"]),
        d["customer_zip_code_prefix"].map(centroids["lng"]),
        d["seller_zip_code_prefix"].map(centroids["lat"]),
        d["seller_zip_code_prefix"].map(centroids["lng"]),
    )
    return {
        "recency_days": float((end - start).days),
        "avg_review_given": float(rev["review_score"].mean()),
        "seller_review_mean": float(rev["review_score"].mean()),
        "distance_km": float(np.nanmedian(dist)),
    }


def pair_features(pairs, cust_f, cust_geo_f, prod_f, seller_f, centroids, fill_stats) -> pd.DataFrame:
    """Feature matrix for (customer_unique_id, product_id) pairs.

    Cold customers (no feature-window history) get has_history=0 with neutral
    fills for behavioural features; that is the serving-time reality for ~97%
    of Olist customers and is reported, not hidden. Geography comes from the
    static customer_geo lookup, so distance/state features survive cold-start.
    Distance is customer zip centroid -> the product's main feature-window
    seller's zip centroid."""
    df = pairs.copy()
    df = df.join(cust_f.add_prefix("c_"), on="customer_unique_id")
    df["has_history"] = df["c_frequency"].notna().astype(int)
    df = df.join(cust_geo_f.add_prefix("g_"), on="customer_unique_id")
    df = df.join(prod_f.add_prefix("p_"), on="product_id")
    df = df.join(seller_f.add_prefix("s_"), on="p_main_seller_id")

    df["category_match"] = (df["c_preferred_category"] == df["p_category"]).astype(int)
    df["price_delta"] = df["p_median_price_w"] - df["c_median_item_price"]
    df["same_state"] = (df["g_customer_state"] == df["s_seller_state"]).astype(int)

    df["_c_lat"] = df["g_customer_zip_code_prefix"].map(centroids["lat"])
    df["_c_lng"] = df["g_customer_zip_code_prefix"].map(centroids["lng"])
    df["_s_lat"] = df["s_seller_zip_code_prefix"].map(centroids["lat"])
    df["_s_lng"] = df["s_seller_zip_code_prefix"].map(centroids["lng"])
    df["distance_km"] = haversine_km(df["_c_lat"], df["_c_lng"], df["_s_lat"], df["_s_lng"])

    numeric = [
        "c_frequency", "c_monetary", "c_recency_days", "c_avg_order_value",
        "c_median_item_price", "c_avg_installments", "c_avg_review_given",
        "p_popularity", "p_median_price_w", "p_freight_ratio", "p_review_mean",
        "p_price_band", "p_has_sales", "p_category_satisfaction_rate",
        "p_product_photos_qty", "p_product_description_lenght", "p_product_weight_g",
        "s_seller_order_count", "s_seller_review_mean",
        "category_match", "price_delta", "same_state", "distance_km", "has_history",
    ]
    fills = {c: 0.0 for c in numeric}
    fills.update({
        "c_recency_days": fill_stats["recency_days"],          # cold = maximally stale
        "c_avg_review_given": fill_stats["avg_review_given"],  # 0 isn't a valid score
        "distance_km": fill_stats["distance_km"],
        "price_delta": 0.0,
        "s_seller_review_mean": fill_stats["seller_review_mean"],
    })
    out = df[["customer_unique_id", "product_id"]].copy()
    for c in numeric:
        out[c] = df[c].astype(float).fillna(fills[c])
    out["p_category"] = df["p_category"].fillna("unknown")
    out["c_region"] = df["g_region"].fillna("unknown")
    return out


def sample_negatives(positives: pd.DataFrame, product_ids: np.ndarray, ratio: int, seed: int) -> pd.DataFrame:
    """Uniform-random negatives: `ratio` non-purchased products per positive pair.
    (Preliminary protocol for notebook 02's feature selection; Step 4 mixes in
    Stage-1 hard negatives per the evaluation plan.)"""
    rng = np.random.default_rng(seed)
    neg = pd.DataFrame({
        "customer_unique_id": np.repeat(positives["customer_unique_id"].to_numpy(), ratio),
        "product_id": rng.choice(product_ids, size=ratio * len(positives)),
    })
    neg = neg.merge(positives.assign(_pos=1), how="left", on=["customer_unique_id", "product_id"])
    neg = neg[neg["_pos"].isna()].drop(columns="_pos")
    return neg.drop_duplicates()
