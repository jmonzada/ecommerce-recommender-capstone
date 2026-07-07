"""LLM-generated recommendation explanations (capstone Step 9: Use of Generative AI).

Turns a recommendation into a customer-facing "why you're seeing this" blurb:
the tuned XGBoost ranker's per-feature SHAP attributions (xgboost's
pred_contribs, i.e. TreeSHAP - the same attribution method as notebook 04)
are translated into plain-English signals and handed to Claude, which writes
one grounded sentence per item.

Cost/latency discipline: responses are cached in models/explanations_cache.json
(committed), so serving stays useful offline and repeated requests never
re-bill. A live API call happens only when a pair is missing from the cache
AND an ANTHROPIC_API_KEY is available (.env, never committed) - otherwise the
item is served without an explanation. No customer identifiers are ever sent
to the API; prompts carry only aggregated signals.

Pre-warm the cache for the demo customers (also saves the prompt and raw
output examples to docs/llm/ as Step 9 evidence):

    python -m src.llm_explain
"""

import json
import os
import threading

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data import REPO_ROOT

CACHE_PATH = REPO_ROOT / "models" / "explanations_cache.json"
_CACHE_LOCK = threading.Lock()  # FastAPI serves sync endpoints from a threadpool
LLM_DOCS_DIR = REPO_ROOT / "docs" / "llm"
MODEL = "claude-opus-4-8"
N_TOP_FEATURES = 5   # SHAP signals quoted per item
EXPLAIN_TOP_N = 3    # items explained per /recommend?explain=true response

FEATURE_GLOSS = {
    "c_frequency": "how many orders the shopper has placed before",
    "c_monetary": "the shopper's total spend so far",
    "c_recency_days": "how long since the shopper's last order",
    "c_avg_order_value": "the shopper's typical order value",
    "c_median_item_price": "the price range the shopper usually buys in",
    "c_avg_installments": "how often the shopper pays in installments",
    "c_avg_review_given": "how the shopper tends to rate purchases",
    "p_popularity": "how many marketplace customers bought this product",
    "p_median_price_w": "the product's price",
    "p_freight_ratio": "shipping cost relative to the product's price",
    "p_review_mean": "the product's average review score",
    "p_price_band": "the product's price tier on the marketplace",
    "p_has_sales": "whether the product has a sales track record",
    "p_category_satisfaction_rate": "how satisfied buyers are with this product's category",
    "p_product_photos_qty": "how many photos the listing has",
    "p_product_description_lenght": "how detailed the product description is",
    "p_product_weight_g": "the product's weight",
    "s_seller_order_count": "the seller's sales volume",
    "s_seller_review_mean": "the seller's average rating",
    "category_match": "the product is in a category the shopper bought before",
    "price_delta": "the price compared with what the shopper usually spends",
    "same_state": "the seller is in the shopper's state",
    "distance_km": "the shipping distance between shopper and seller",
    "has_history": "the shopper has purchase history on the marketplace",
    "cf_signal": "customers with similar purchases also bought this product",
    "has_cf_signal": "a similar-customers signal exists for this pairing",
    "p_category_freq": "how common the product's category is on the marketplace",
}


# binary/one-hot signals read differently when the feature is absent (value 0);
# quoting the positive gloss for an absent feature would put invented facts in
# front of the LLM (e.g. "the shopper is from the Northeast" for a visitor
# whose region is unknown)
NEGATED_GLOSS = {
    "category_match": "the product is outside the categories the shopper bought before",
    "same_state": "the seller is in a different state from the shopper",
    "has_history": "the shopper has no purchase history on the marketplace",
    "has_cf_signal": "no similar-customers signal exists for this pairing",
    "cf_signal": "no similar-purchases signal links the shopper to this product",
    "p_has_sales": "the product has no prior sales in the training window",
}


def feature_gloss(name: str) -> str:
    if name.startswith("region_"):
        return f"the shopper is from the {name.removeprefix('region_')} region"
    return FEATURE_GLOSS[name]


def signal_gloss(name: str, value: float, region_known: bool = True) -> str:
    """Value-aware gloss: negate binary/one-hot signals when the feature is
    absent, so the prompt never asserts something untrue about the pairing.
    When the shopper's region is unknown, every region one-hot is 0 because
    nothing is known - "not from X" would still overclaim, so those glosses
    collapse to "region is unknown"."""
    if name.startswith("region_"):
        if not region_known:
            return "the shopper's region is unknown to the marketplace"
        if value < 0.5:
            return f"the shopper is not from the {name.removeprefix('region_')} region"
        return feature_gloss(name)
    if name in NEGATED_GLOSS and value <= 0:
        return NEGATED_GLOSS[name]
    return feature_gloss(name)


def top_contributions(ranker, x: pd.DataFrame, n: int = N_TOP_FEATURES) -> list[list[dict]]:
    """Per-row TreeSHAP attributions for the fitted XGBClassifier, strongest
    first. Returns one list of {feature, value, contribution} dicts per row
    of `x` (the bias term is dropped)."""
    contribs = ranker.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
    out = []
    for r, row in enumerate(contribs):
        vals = row[:-1]  # last column is the bias term
        order = np.argsort(-np.abs(vals))[:n]
        out.append([
            {"feature": x.columns[i], "value": float(x.iat[r, i]),
             "contribution": float(vals[i])}
            for i in order
        ])
    return out


def _fmt(v: float | None, suffix: str = "") -> str:
    return "unknown" if v is None or v != v else f"{v:.2f}{suffix}"


def shopper_line(route: str, region: str | None, n_prior: float | None,
                 preferred_category: str | None) -> str:
    if route == "two_stage" and n_prior:
        line = f"a repeat customer with {int(n_prior)} previous order(s)"
        if isinstance(preferred_category, str) and preferred_category:
            line += f", mostly in the '{preferred_category}' category"
        return line
    if route == "regional_popularity" and region:
        return (f"a customer from Brazil's {region} region with no purchase "
                "history in the training window")
    return "a first-time visitor with no purchase history"


def build_prompt(item: dict, shopper: str, signals: list[dict],
                 region_known: bool = True) -> str:
    lines, seen = [], set()
    for s in signals:
        gloss = signal_gloss(s["feature"], s["value"], region_known)
        if gloss in seen:  # several absent region one-hots collapse to one line
            continue
        seen.add(gloss)
        direction = ("pushed the recommendation up" if s["contribution"] > 0
                     else "pulled it down")
        lines.append(f"- {gloss} ({direction})")
    signal_lines = "\n".join(lines)
    return f"""You write short, friendly product-recommendation explanations for shoppers
on Olist, a Brazilian online marketplace.

Product being recommended:
- category: {item.get("category", "unknown")}
- price: R$ {_fmt(item.get("price_brl"))}
- average review score: {_fmt(item.get("review_mean"), "/5")}
- bought by {int(item.get("popularity", 0))} customers in the training window

The shopper is {shopper}.

The recommendation model's strongest signals for this pairing, from SHAP
feature attributions, strongest first:
{signal_lines}

Write ONE sentence (under 35 words) telling the shopper why they're seeing
this product. Rules:
- Ground it only in the signals and context above; invent nothing.
- Plain language: no scores, no model or feature names, never the words
  "algorithm", "model", or "SHAP".
- Only mention a "pulled it down" signal if you can reframe it honestly
  (e.g. "a bit above your usual spend").
- Address the shopper as "you". Output only the sentence."""


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        import logging  # a corrupt cache degrades to "nothing cached", never a 500
        logging.getLogger(__name__).warning("explanations cache unreadable - treating as empty")
        return {}


def save_cache(cache: dict) -> None:
    tmp = CACHE_PATH.with_name(CACHE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, CACHE_PATH)  # atomic: readers never see a half-written file


def cache_key(customer_id: str, product_id: str) -> str:
    return f"{customer_id}:{product_id}"


def _signal_note(s: dict) -> str:
    """Auditable signal record for the cache/evidence files: name, feature
    value, and signed SHAP contribution."""
    return (f"{s['feature']}={s['value']:.4g} "
            f"({'+' if s['contribution'] > 0 else ''}{s['contribution']:.2f})")


def have_api_key() -> bool:
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass  # cache-only environments don't need dotenv
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _live_explanation(prompt: str) -> str:
    from anthropic import Anthropic  # lazy: cache-only serving works without the SDK
    message = Anthropic().messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in message.content if b.type == "text").strip()


def explain_recommendations(art: dict, ranker, feature_columns: list[str],
                            customer_id: str, top: pd.DataFrame, route: str,
                            max_items: int = EXPLAIN_TOP_N,
                            allow_live: bool | None = None) -> list[dict | None]:
    """Explanations for the first `max_items` rows of a recommend_two_stage
    frame. Cache-first; live Claude calls only when allowed and a key exists.
    Returns one {text, source} dict (or None) per explained row."""
    from src.pipeline import ranker_matrix

    if allow_live is None:
        allow_live = have_api_key()
    sub = top.head(max_items)
    pairs = pd.DataFrame({"customer_unique_id": customer_id,
                          "product_id": sub["product_id"].to_numpy()})
    signals = top_contributions(ranker, ranker_matrix(pairs, art, columns=feature_columns))

    cust = art["cust_fw"]
    region = art["geo_region"].get(customer_id)
    n_prior = cust["frequency"].get(customer_id) if customer_id in cust.index else None
    pref = (cust["preferred_category"].get(customer_id)
            if "preferred_category" in cust.columns and customer_id in cust.index else None)
    shopper = shopper_line(route, region, n_prior, pref)
    review_mean = art["prod_fw"]["review_mean"]

    with _CACHE_LOCK:
        cache = load_cache()
    new_entries, results = {}, []
    for row, sig in zip(sub.itertuples(), signals):
        key = cache_key(customer_id, row.product_id)
        if key in cache:
            results.append({"text": cache[key]["explanation"], "source": "cache"})
            continue
        if not allow_live:
            results.append(None)
            continue
        item = {"category": row.category, "price_brl": row.price_brl,
                "popularity": row.popularity,
                "review_mean": float(review_mean.get(row.product_id, float("nan")))}
        try:
            text = _live_explanation(
                build_prompt(item, shopper, sig, region_known=region is not None))
        except Exception as exc:  # explanations are progressive enhancement -
            import logging       # an LLM outage must not fail the request
            logging.getLogger(__name__).warning("live explanation failed: %s", exc)
            results.append(None)
            continue
        new_entries[key] = {
            "explanation": text, "model": MODEL, "route": route,
            "signals": [_signal_note(s) for s in sig],
            "context": {"category": str(row.category),
                        "price_brl": None if row.price_brl != row.price_brl
                        else float(row.price_brl),
                        "popularity": int(row.popularity),
                        "review_mean": round(item["review_mean"], 2)},
        }
        results.append({"text": text, "source": "live"})
    if new_entries:
        with _CACHE_LOCK:  # merge-then-replace: concurrent writers keep each
            save_cache({**load_cache(), **new_entries})  # other's paid blurbs
    return results


def prewarm() -> None:
    """Generate and cache explanations for the demo customers (one per route),
    saving prompt + output evidence to docs/llm/. Idempotent via the cache."""
    import joblib

    from src.features import load_clean_orders
    from src.pipeline import build_artifacts, recommend_two_stage

    if not have_api_key():
        raise SystemExit("ANTHROPIC_API_KEY missing (.env) - cannot pre-warm the cache")

    ranker = joblib.load(REPO_ROOT / "models" / "ranker_xgboost.joblib")
    feature_columns = json.loads(
        (REPO_ROOT / "models" / "feature_columns.json").read_text())
    art = build_artifacts()

    # one demo customer per route: the same repeat buyer the demo media uses,
    # a real customer with a region but no feature-window history, and a
    # completely unknown visitor
    delivered, _ = load_clean_orders()
    counts = delivered.groupby("customer_unique_id")["order_id"].nunique()
    repeat_id = counts[counts >= 2].index[0]
    known_users = set(art["im_fw"].user_index)
    cold_id = next(c for c in art["geo"].index if c not in known_users)
    demo_customers = [repeat_id, cold_id, "first-time-visitor"]

    LLM_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    example_lines = [
        "# LLM recommendation explanations - generated examples", "",
        f"Model: `{MODEL}`. Each pairing lists the item context handed to the "
        "model and its top SHAP signals (feature=value, signed contribution), "
        "so every blurb can be checked against exactly what the prompt "
        "contained (Step 9 verification evidence).", "",
        "## Preserved: before the value-aware gloss fix", "",
        "The first generation run glossed one-hot signals by name and ignored "
        "their value. For the unknown visitor (route `global_popularity`, item "
        "`bb50f2e236e5...`), whose region one-hots are all 0, the prompt "
        'asserted "the shopper is from the Northeast region" and the blurb '
        "came back:", "",
        "> We're showing you this well-photographed health & beauty product "
        "because its detailed listing, price, and popularity with shoppers in "
        "your Northeast region make it a strong match for you.", "",
        "That region claim is invented - the system doesn't know the "
        "visitor's region. `test_signal_gloss_is_value_aware` pins the fix; "
        "unknown-region visitors now get a 'region is unknown' gloss instead "
        "of a negated one.", "",
    ]
    prompt_saved = False
    for cid in demo_customers:
        top, route = recommend_two_stage(art, ranker, feature_columns, cid, k=EXPLAIN_TOP_N)
        # save one full prompt (the first pairing) as the before/after evidence
        if not prompt_saved:
            pairs = pd.DataFrame({"customer_unique_id": cid,
                                  "product_id": top["product_id"].to_numpy()})
            from src.pipeline import ranker_matrix
            sig0 = top_contributions(ranker, ranker_matrix(pairs, art, columns=feature_columns))[0]
            row0 = next(top.itertuples())
            item0 = {"category": row0.category, "price_brl": row0.price_brl,
                     "popularity": row0.popularity,
                     "review_mean": float(art["prod_fw"]["review_mean"].get(row0.product_id, float("nan")))}
            shopper0 = shopper_line(route, art["geo_region"].get(cid),
                                    art["cust_fw"]["frequency"].get(cid) if cid in art["cust_fw"].index else None,
                                    art["cust_fw"]["preferred_category"].get(cid) if cid in art["cust_fw"].index else None)
            (LLM_DOCS_DIR / "explanation_prompt_example.md").write_text(
                build_prompt(item0, shopper0, sig0,
                             region_known=art["geo_region"].get(cid) is not None),
                encoding="utf-8")
            prompt_saved = True

        results = explain_recommendations(art, ranker, feature_columns, cid, top, route)
        example_lines.append(f"## route: {route}")
        example_lines.append("")
        cache = load_cache()
        for row, res in zip(top.itertuples(), results):
            if res is None:  # a transient API failure must not abort the run
                example_lines += [f"**{row.product_id[:12]}… / {row.category}** "
                                  "- live call failed, not cached", ""]
                continue
            entry = cache[cache_key(cid, row.product_id)]
            ctx = entry.get("context", {})
            example_lines += [
                f"**{row.product_id[:12]}… / {row.category} / R$ {_fmt(row.price_brl)}**",
                f"- context given to the model: review {ctx.get('review_mean', '?')}/5, "
                f"bought by {ctx.get('popularity', '?')} customers in the window",
                f"- signals: {', '.join(entry['signals'])}",
                f"- explanation ({res['source']}): {entry['explanation']}", "",
            ]
        print(f"{route}: {sum(r is not None for r in results)} explanations cached")

    (LLM_DOCS_DIR / "explanation_examples.md").write_text(
        "\n".join(example_lines), encoding="utf-8")
    print(f"wrote {LLM_DOCS_DIR / 'explanation_examples.md'}")
    print(f"cache now holds {len(load_cache())} explanations at {CACHE_PATH}")


if __name__ == "__main__":
    prewarm()
