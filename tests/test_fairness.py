"""Tests for src/fairness.py - group table semantics, suppression, exposure."""

import numpy as np
import pandas as pd

from src.fairness import (
    MIN_GROUP_N,
    exposure_gini,
    group_metric_table,
    lorenz_points,
    rerank_with_popularity_penalty,
)


def test_group_table_suppresses_small_groups_and_reports_n():
    rng = np.random.default_rng(42)
    n_big = 200
    y = np.concatenate([rng.integers(0, 2, n_big), np.array([1, 0, 1])])
    s = np.concatenate([rng.random(n_big), np.array([0.9, 0.1, 0.8])])
    g = np.array(["big"] * n_big + ["tiny"] * 3)
    t = group_metric_table(y, s, g, tau=0.5)
    assert t.loc["tiny", "n"] == 3  # n always reported
    assert np.isnan(t.loc["tiny"].get("selection_rate", np.nan))  # metrics suppressed
    assert t.loc["big", "n"] >= MIN_GROUP_N
    assert 0 <= t.loc["big", "tpr"] <= 1
    assert t.loc["big", "tpr_lo"] <= t.loc["big", "tpr"] <= t.loc["big", "tpr_hi"]


def test_group_table_known_tpr_and_dp():
    # group A: both positives above tau; group B: neither
    y = np.array([1, 1, 0, 0] * 20 + [1, 1, 0, 0] * 20)
    s = np.array([0.9, 0.8, 0.1, 0.2] * 20 + [0.3, 0.2, 0.1, 0.2] * 20)
    g = np.array(["A"] * 80 + ["B"] * 80)
    t = group_metric_table(y, s, g, tau=0.5)
    assert t.loc["A", "tpr"] == 1.0
    assert t.loc["B", "tpr"] == 0.0
    assert abs(t.attrs["dp_difference"] - 0.5) < 1e-9
    assert t.attrs["di_ratio"] == 0.0


def test_exposure_gini_bounds():
    assert exposure_gini(np.ones(100)) < 0.01          # perfectly equal
    concentrated = np.zeros(100)
    concentrated[0] = 500
    assert exposure_gini(concentrated) > 0.95          # all slots on one item
    xs, ys = lorenz_points(np.arange(100))
    assert ys[0] <= ys[-1] and abs(ys[-1] - 1.0) < 1e-9


def test_popularity_penalty_rerank_demotes_head_items():
    scored = pd.DataFrame({
        "customer_unique_id": ["u"] * 3,
        "product_id": ["head", "mid", "tail"],
        "p": [0.60, 0.55, 0.50],
        "pop_pct": [1.0, 0.5, 0.0],
    })
    top0 = rerank_with_popularity_penalty(scored, lam=0.0, k=3)["u"]
    assert list(top0) == ["head", "mid", "tail"]
    top = rerank_with_popularity_penalty(scored, lam=0.2, k=3)["u"]
    assert list(top) == ["tail", "mid", "head"]  # penalty flips the order
