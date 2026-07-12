"""FastAPI serving app (capstone Step 8: local deployment).

Serves the final Step 4 pipeline - routed hybrid/popularity candidates
re-ranked by the tuned XGBoost - with cold-start handling built into the
router. Artifacts are rebuilt from the committed data and configs at startup
(a few seconds; GET /health reports readiness). Run with:

    uvicorn app.main:app --port 8000
"""

import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from src import llm_explain
from src.data import REPO_ROOT
from src.pipeline import build_artifacts, recommend_two_stage

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["ranker"] = joblib.load(REPO_ROOT / "models" / "ranker_xgboost.joblib")
    STATE["features"] = json.loads(
        (REPO_ROOT / "models" / "feature_columns.json").read_text()
    )
    STATE["art"] = build_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="Olist product recommender",
    description="Two-stage recommender (hybrid candidates + XGBoost re-rank) "
                "with cold-start routing. AIM capstone, Step 8.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts_loaded": "art" in STATE}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, k: int = Query(default=10, ge=1, le=50),
              explain: bool = Query(default=False)) -> dict:
    if "art" not in STATE:
        raise HTTPException(status_code=503, detail="artifacts still loading")
    top, route = recommend_two_stage(
        STATE["art"], STATE["ranker"], STATE["features"], customer_id, k=k
    )
    items = [
        {
            "product_id": r.product_id,
            "score": round(float(r.score), 4),
            "category": r.category,
            "price_brl": None if r.price_brl != r.price_brl else float(r.price_brl),
            "popularity": int(r.popularity),
        }
        for r in top.itertuples()
    ]
    if explain:
        # Step 9: cache-first Claude blurbs for the top items; live calls only
        # when an API key is configured, otherwise unexplained items get null
        blurbs = llm_explain.explain_recommendations(
            STATE["art"], STATE["ranker"], STATE["features"],
            customer_id, top, route,
        )
        for item, blurb in zip(items, blurbs):
            item["explanation"] = None if blurb is None else blurb["text"]
            item["explanation_source"] = None if blurb is None else blurb["source"]
    return {"customer_id": customer_id, "route": route, "k": k, "items": items}


DEMO_HTML = """<!doctype html>
<html><head><title>Olist recommender demo</title><style>
body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; }
input { width: 24rem; padding: .4rem; } button { padding: .4rem 1rem; }
table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
td, th { border: 1px solid #ccc; padding: .35rem .6rem; font-size: .9rem; text-align: left; }
.route { margin-top: .8rem; color: #444; }
.why td { border-top: none; color: #2a6f2a; font-style: italic; }
label { margin-left: .8rem; font-size: .9rem; }
</style></head><body>
<h2>Olist product recommender</h2>
<p>Paste a <code>customer_unique_id</code> (or anything, to see the cold-start
fallback) and get the two-stage pipeline's top picks.</p>
<input id="cid" placeholder="customer_unique_id" value="">
<button onclick="go()">Recommend</button>
<label><input type="checkbox" id="explain"> explain top picks (LLM)</label>
<div class="route" id="route"></div>
<table id="out"></table>
<script>
async function go() {
  const cid = document.getElementById('cid').value || 'unknown-visitor';
  const explain = document.getElementById('explain').checked;
  const r = await fetch('/recommend/' + encodeURIComponent(cid)
                        + '?k=10&explain=' + explain);
  const data = await r.json();
  document.getElementById('route').textContent = 'route: ' + data.route;
  const rows = data.items.map((it, i) =>
    `<tr><td>${i + 1}</td><td>${it.product_id.slice(0, 12)}…</td>` +
    `<td>${it.category}</td><td>${it.price_brl ?? ' - '}</td>` +
    `<td>${it.score}</td></tr>` +
    (it.explanation
      ? `<tr class="why"><td></td><td colspan="4">${it.explanation}</td></tr>` : '')
  ).join('');
  document.getElementById('out').innerHTML =
    '<tr><th>#</th><th>product</th><th>category</th><th>price (BRL)</th><th>score</th></tr>' + rows;
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return DEMO_HTML
