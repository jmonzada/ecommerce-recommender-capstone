"""Data access helpers for the Olist capstone.

Covers rubric Step 2 (data collection) groundwork: canonical paths to the raw
Olist tables and derivation of the committed zip-prefix centroid lookup, which
stands in for the 58 MB geolocation file that is deliberately not committed
(see data/README.md).
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

RAW_TABLES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

GEO_CENTROIDS_PATH = PROCESSED_DIR / "geo_zip_centroids.csv"

# IBGE macro-regions. Used for EDA rollups and as the socioeconomic proxy
# grouping in the fairness audit (see reports/final_report.md).
BR_STATE_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North",
    "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast",
    "MA": "Northeast", "PB": "Northeast", "PE": "Northeast",
    "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Center-West", "GO": "Center-West", "MT": "Center-West", "MS": "Center-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}


def load_table(name: str) -> pd.DataFrame:
    """Load one raw Olist table by short name (see RAW_TABLES)."""
    return pd.read_csv(RAW_DIR / RAW_TABLES[name])


def derive_geo_centroids(out_path: Path = GEO_CENTROIDS_PATH) -> pd.DataFrame:
    """Collapse the geolocation table to one mean lat/lng per zip prefix.

    The raw file has ~1M rows (many points per zip prefix); the centroid
    lookup is what the haversine distance feature actually needs, and it is
    small enough to commit. Requires the raw geolocation CSV to be present
    (Kaggle download instructions in data/README.md).
    """
    geo = load_table("geolocation")
    centroids = (
        geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            n_points=("geolocation_lat", "size"),
        )
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_csv(out_path, index=False)
    print(f"Wrote {len(centroids)} zip-prefix centroids to {out_path}")
    return centroids


if __name__ == "__main__":
    derive_geo_centroids()
