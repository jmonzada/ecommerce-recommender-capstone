"""API smoke tests (capstone Step 8). One artifact build per session."""

import pytest
from fastapi.testclient import TestClient

from src.data import load_table


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:  # context manager runs the lifespan (artifact build)
        yield c


@pytest.fixture(scope="module")
def warm_customer():
    # any customer that exists in the data works; the router decides the route
    return load_table("customers")["customer_unique_id"].iloc[0]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "artifacts_loaded": True}


def test_recommend_known_customer(client, warm_customer):
    r = client.get(f"/recommend/{warm_customer}?k=5")
    assert r.status_code == 200
    body = r.json()
    assert body["route"] in {"two_stage", "regional_popularity", "global_popularity"}
    assert len(body["items"]) == 5
    scores = [it["score"] for it in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(set(it) == {"product_id", "score", "category", "price_brl", "popularity"}
               for it in body["items"])


def test_recommend_unknown_customer_falls_back(client):
    r = client.get("/recommend/definitely-not-a-real-customer?k=3")
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "global_popularity"
    assert len(body["items"]) == 3


def test_recommend_explain_serves_from_cache(client, warm_customer, tmp_path, monkeypatch):
    from src import llm_explain

    base = client.get(f"/recommend/{warm_customer}?k=5").json()
    monkeypatch.setattr(llm_explain, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(llm_explain, "have_api_key", lambda: False)  # never call the API
    llm_explain.save_cache({
        llm_explain.cache_key(warm_customer, it["product_id"]):
            {"explanation": f"cached blurb {i}"}
        for i, it in enumerate(base["items"][:2])
    })

    body = client.get(f"/recommend/{warm_customer}?k=5&explain=true").json()
    assert body["items"][0]["explanation"] == "cached blurb 0"
    assert body["items"][0]["explanation_source"] == "cache"
    assert body["items"][1]["explanation"] == "cached blurb 1"
    # third item is uncached and live calls are disabled -> explicit null
    assert body["items"][2]["explanation"] is None
    # beyond EXPLAIN_TOP_N the field is absent entirely
    assert "explanation" not in body["items"][llm_explain.EXPLAIN_TOP_N]


def test_recommend_without_explain_has_no_explanation_field(client, warm_customer):
    body = client.get(f"/recommend/{warm_customer}?k=3").json()
    assert all("explanation" not in it for it in body["items"])


def test_k_validation(client):
    assert client.get("/recommend/x?k=0").status_code == 422
    assert client.get("/recommend/x?k=99").status_code == 422


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Olist product recommender" in r.text
