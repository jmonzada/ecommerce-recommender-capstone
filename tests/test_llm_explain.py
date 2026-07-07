"""LLM explanation module tests (capstone Step 9). No network calls."""

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src import llm_explain
from src.data import REPO_ROOT


def test_feature_gloss_covers_every_training_column():
    cols = json.loads((REPO_ROOT / "models" / "feature_columns.json").read_text())
    for col in cols:
        assert llm_explain.feature_gloss(col), col


def test_top_contributions_sorted_by_magnitude():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = (x["a"] + 0.5 * x["b"] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42).fit(x, y)

    per_row = llm_explain.top_contributions(model, x.head(3), n=3)
    assert len(per_row) == 3
    for signals in per_row:
        assert len(signals) == 3
        mags = [abs(s["contribution"]) for s in signals]
        assert mags == sorted(mags, reverse=True)
        assert all(s["feature"] in list("abcd") for s in signals)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_explain, "CACHE_PATH", tmp_path / "cache.json")
    assert llm_explain.load_cache() == {}
    llm_explain.save_cache({"c:p": {"explanation": "hi"}})
    assert llm_explain.load_cache()["c:p"]["explanation"] == "hi"


def test_corrupt_cache_degrades_to_empty(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    path.write_text("{truncated", encoding="utf-8")
    monkeypatch.setattr(llm_explain, "CACHE_PATH", path)
    assert llm_explain.load_cache() == {}  # never a 500 on the request path


def test_prompt_is_grounded_in_signals():
    signals = [{"feature": "cf_signal", "value": 0.4, "contribution": 1.2},
               {"feature": "price_delta", "value": -12.0, "contribution": -0.3}]
    item = {"category": "bed_bath_table", "price_brl": 89.9,
            "review_mean": 4.2, "popularity": 57}
    prompt = llm_explain.build_prompt(
        item, llm_explain.shopper_line("two_stage", "Southeast", 2, "bed_bath_table"),
        signals)
    assert "customers with similar purchases also bought this product" in prompt
    assert "pulled it down" in prompt
    assert "R$ 89.90" in prompt
    assert "repeat customer with 2 previous order(s)" in prompt
    assert "cf_signal" not in prompt  # raw feature names never reach the prompt


def test_signal_gloss_is_value_aware():
    # absent one-hot/binary features must be negated, not asserted
    assert "not from the Northeast" in llm_explain.signal_gloss("region_Northeast", 0.0)
    assert "is from the Northeast" in llm_explain.signal_gloss("region_Northeast", 1.0)
    assert "outside the categories" in llm_explain.signal_gloss("category_match", 0.0)
    assert "bought before" in llm_explain.signal_gloss("category_match", 1.0)
    assert "no similar-purchases signal" in llm_explain.signal_gloss("cf_signal", 0.0)
    assert "also bought this product" in llm_explain.signal_gloss("cf_signal", 0.37)
    # continuous features pass through regardless of value
    assert llm_explain.signal_gloss("distance_km", 0.0) == llm_explain.feature_gloss("distance_km")
    # unknown region: every region one-hot is 0 because NOTHING is known, so
    # "not from X" would still overclaim
    assert "region is unknown" in llm_explain.signal_gloss(
        "region_Southeast", 0.0, region_known=False)


def test_prompt_collapses_unknown_region_signals():
    signals = [{"feature": "region_Southeast", "value": 0.0, "contribution": 0.5},
               {"feature": "region_Northeast", "value": 0.0, "contribution": -0.2},
               {"feature": "p_popularity", "value": 40.0, "contribution": 0.8}]
    item = {"category": "toys", "price_brl": 50.0, "review_mean": 4.0, "popularity": 40}
    prompt = llm_explain.build_prompt(
        item, "a first-time visitor with no purchase history", signals,
        region_known=False)
    assert prompt.count("region is unknown") == 1  # deduped to one line
    assert "not from the" not in prompt


def test_shopper_line_routes():
    assert "repeat customer" in llm_explain.shopper_line("two_stage", None, 3, None)
    assert "Northeast region" in llm_explain.shopper_line(
        "regional_popularity", "Northeast", None, None)
    assert "first-time visitor" in llm_explain.shopper_line(
        "global_popularity", None, None, None)
    # NaN preferred category must not leak into the prompt
    assert "nan" not in llm_explain.shopper_line("two_stage", None, 2, float("nan"))
