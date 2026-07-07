# Deployment Guide (Step 8)

## Run locally

```bash
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
uvicorn app.main:app --port 8000
```

Startup rebuilds the serving artefacts from the committed data and
`models/chosen_config.yaml` (a few seconds). Then:

- `GET /` — interactive demo page (paste a `customer_unique_id`, get top-10)
- `GET /health` — readiness probe (`artifacts_loaded: true` once serving)
- `GET /recommend/{customer_unique_id}?k=10` — the two-stage pipeline; unknown
  customers get the cold-start route automatically (`route` names which path
  served the request: `two_stage` / `regional_popularity` / `global_popularity`)

Demo: ![demo](media/demo.gif)

## Docker

```bash
docker build -t olist-recommender .
docker run -p 8000:8000 olist-recommender
```

The image bakes in the raw data, configs, and trained artefacts, so a container
serves identically to the local run — no external services needed.

## Reproducibility & experiment tracking

- Environment: `requirements.txt` (pinned exactly before submission) + this
  Dockerfile (python:3.11-slim).
- Config-driven runs: candidate/ranker settings live in `configs/models.yaml`;
  the evaluation windows in `configs/windows.yaml`; the winning configuration is
  frozen to `models/chosen_config.yaml`.
- Every Step 4 training/evaluation run is logged to MLflow (local SQLite store,
  `mlruns.db`, gitignored). Inspect with `mlflow ui --backend-store-uri sqlite:///mlruns.db`.

## Monitoring plan

Signals to watch in production, in order of alarm value:

1. **Routing-share drift** — the share of requests served by each route
   (`two_stage` vs `regional_popularity` vs `global_popularity`). A shift
   means the customer mix changed; the 98.5% cold share is the baseline.
2. **Score distribution drift** — weekly PSI on the ranker's score histogram
   per route. Drift indicates the feature pipeline or demand changed.
3. **Online hit-rate telemetry** — clicks/purchases on recommended items vs
   the offline HitRate@10 baselines (0.0146 overall, per-region values in
   `models/fairness_metrics.json`). This is where the offline proxies get
   validated or falsified.
4. **Exposure metrics** — recompute the exposure Gini and small-seller share
   monthly (the fairness audit's provider-side numbers are the baseline);
   popularity feedback loops show up here first.
5. **Operational** — p95 latency of `/recommend` and error rate of 5xx.

## Versioning & rollback

- Model artefacts (`models/*.joblib`, `*.npy`, `*.npz`, `chosen_config.yaml`)
  are committed alongside the code that produced them, so **every git tag is a
  deployable snapshot**. Tag releases (`v0.1.0`, ...) after each retraining.
- Rollback = redeploy the previous tag (`git checkout <tag>` + `docker build`,
  or re-pull the previous image). No database migrations exist; rollback is
  side-effect free.
- Retraining runs re-execute notebook 03 (seeded), which refreshes the
  artefacts and MLflow records; the paired-CI evaluation in that notebook is
  the promotion gate — a new model ships only if the end-to-end paired
  difference vs the incumbent is positive.

## Known limitations

- Single-process, in-memory artefacts; horizontal scaling would move artefact
  loading behind a shared store.
- The reserved-slot exploration mitigation from Step 5 is NOT enabled in the
  serving path — it requires fresh-window validation first (see the report).
