"""Tests for src/features.py — window discipline, cleaning rules, pair features."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    assert_max_ts,
    haversine_km,
    load_clean_items,
    load_clean_orders,
    load_clean_products,
    load_clean_reviews,
    load_windows,
    sample_negatives,
)


def test_haversine_sp_to_rio():
    # São Paulo (-23.55, -46.63) to Rio (-22.91, -43.17): great-circle ~357 km
    d = haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 340 < d < 380


def test_windows_are_ordered_and_half_open():
    w = load_windows()
    assert w["feature_window"][1] <= w["label_window"][0] + pd.Timedelta(days=1)
    assert w["feature_window"][1] <= w["label_window"][0]
    assert w["label_window"][1] <= w["holdout"][0]
    # end bounds are exclusive next-day midnights
    assert w["holdout"][1] == pd.Timestamp("2018-09-01")


def test_assert_max_ts_raises_on_leakage():
    df = pd.DataFrame({"ts": pd.to_datetime(["2018-01-01", "2018-09-15"])})
    with pytest.raises(AssertionError, match="leakage"):
        assert_max_ts(df, "ts", pd.Timestamp("2018-09-01"))
    assert_max_ts(df[df["ts"] < "2018-09-01"], "ts", pd.Timestamp("2018-09-01"))


def test_cleaning_counts_are_reported():
    delivered, oc = load_clean_orders()
    assert oc["dropped_not_delivered"] == oc["orders_raw"] - len(delivered)
    assert (delivered["order_status"] == "delivered").all()
    assert delivered["region"].notna().all()

    reviews, rc = load_clean_reviews()
    assert reviews["order_id"].is_unique
    assert rc["dropped_duplicate_reviews"] > 0

    items, ic = load_clean_items()
    assert ic["item_lines_after_unit_dedupe"] < ic["item_rows_raw"]
    assert not items.duplicated(["order_id", "product_id", "seller_id"]).any()

    products, pc = load_clean_products()
    assert pc["missing_category"] > 0
    assert products["category"].notna().all()


def test_sample_negatives_excludes_positives_and_is_seeded():
    positives = pd.DataFrame({
        "customer_unique_id": ["u1", "u1", "u2"],
        "product_id": ["a", "b", "a"],
    })
    pool = np.array(["a", "b", "c", "d", "e"])
    neg1 = sample_negatives(positives, pool, ratio=4, seed=42)
    neg2 = sample_negatives(positives, pool, ratio=4, seed=42)
    pd.testing.assert_frame_equal(neg1.reset_index(drop=True), neg2.reset_index(drop=True))
    merged = neg1.merge(positives, on=["customer_unique_id", "product_id"], how="inner")
    assert len(merged) == 0
