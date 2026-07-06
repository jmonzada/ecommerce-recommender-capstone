"""Tests for src/data.py — raw table access and the committed centroid lookup."""

import pandas as pd

from src.data import GEO_CENTROIDS_PATH, RAW_DIR, RAW_TABLES, load_table


def test_raw_tables_present():
    # geolocation is deliberately not committed (see data/README.md)
    expected = set(RAW_TABLES) - {"geolocation"}
    missing = [name for name in expected if not (RAW_DIR / RAW_TABLES[name]).exists()]
    assert not missing, f"missing raw tables: {missing}"


def test_load_products():
    df = load_table("products")
    assert not df.empty
    assert "product_id" in df.columns
    assert df["product_id"].is_unique


def test_geo_centroids_lookup():
    assert GEO_CENTROIDS_PATH.exists(), "run `python -m src.data` to derive it"
    centroids = pd.read_csv(GEO_CENTROIDS_PATH)
    assert {"zip_code_prefix", "lat", "lng", "n_points"} <= set(centroids.columns)
    assert centroids["zip_code_prefix"].is_unique
    assert len(centroids) > 10_000
