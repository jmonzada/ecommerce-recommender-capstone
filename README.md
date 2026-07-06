# eCommerce Product Recommender — Olist Marketplace

End-to-end machine learning capstone project: a two-stage product recommendation system built on the [Olist Brazilian eCommerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100k real marketplace orders, 2016–2018).

> Capstone for the Post Graduate Diploma in Artificial Intelligence and Machine Learning (AIM/Emeritus) — Daniel Jethro Monzada.

## Problem

Olist is a marketplace that connects small Brazilian sellers to large storefronts. Roughly 97% of its customers never come back for a second order. This project builds a recommender that targets that gap: surface relevant products to each customer to lift repeat purchases and cross-sell, while giving long-tail sellers fair exposure.

**Task type:** recommendation, decomposed into two stages —

1. **Candidate generation** — popularity baseline, item-item collaborative filtering, content-based similarity, truncated-SVD matrix factorization, and a hybrid blend.
2. **Conversion ranking** — a supervised classifier (Logistic Regression / Random Forest / XGBoost) that scores candidate (customer, product) pairs using engineered behavioural, product, and pair features.

**Technical metrics:** HitRate@10, NDCG@10, catalogue coverage (stage 1); ROC-AUC, PR-AUC, F1 (stage 2).
**Fairness:** recommendation-quality parity across Brazilian regions, seller-exposure equity, popularity-bias measurement — with a mitigation re-ranker and a measured fairness/accuracy trade-off.

## Results

*(Populated as modelling lands — see `reports/final_report.md` for the full analysis.)*

| Model | HitRate@10 | NDCG@10 | Coverage |
|-------|-----------|---------|----------|
| Popularity baseline | – | – | – |
| Item-item CF | – | – | – |
| Content-based | – | – | – |
| Truncated SVD | – | – | – |
| Hybrid | – | – | – |

## Repository layout

```
configs/          # YAML experiment configs
data/raw/         # Olist source CSVs (see data/README.md for licence + download)
data/processed/   # derived artefacts (regenerated; committed: zip-prefix centroid lookup)
docs/             # data dictionary, deployment guide, demo media
notebooks/        # 01 data overview · 02 EDA & features · 03 modeling · 04 explainability & fairness · 05 technical slides
src/              # reusable pipeline code (data, features, models, evaluation, fairness, recommend)
models/           # saved model artefacts + per-run metrics
reports/          # final report + figures
presentations/    # technical + business slide decks
app/              # FastAPI recommendation service + demo page
tests/            # pytest suite
```

## Quickstart

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# data: see data/README.md (Kaggle download instructions)

# run the notebooks in order, or serve the trained model:
uvicorn app.main:app --reload
```

## AI assistance

Generative AI (Anthropic's Claude) was used in this project as a development assistant — code scaffolding, review, and documentation drafting — and as a project feature (LLM-generated recommendation explanations and an LLM-drafted data dictionary, both documented in the final report's *Use of Generative AI* section). All analysis decisions, results, and final content were verified and are owned by the author.

## Licence

Code: MIT (see `LICENSE`). Data: Olist dataset, CC BY-NC-SA 4.0 — see `data/README.md`.
