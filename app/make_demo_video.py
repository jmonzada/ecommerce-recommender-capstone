"""Generate docs/media/demo.mp4 - the combined Step 8 + Step 9 demo video.

Drives the FastAPI app in-process (TestClient) and renders the actual JSON
responses into video frames: health check, all three recommendation routes
(repeat buyer / regional cold-start / unknown visitor), and the Step 9
explain=true flow with the cached Claude blurbs. Reproducible demo media:
every number and every explanation shown comes from a real request.

    python -m app.make_demo_video
"""

import json
import textwrap

import imageio.v2 as imageio
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from src.data import REPO_ROOT
from src.features import load_clean_orders

W, H = 912, 560  # multiples of 16 keep the H.264 encoder from resizing
FPS = 10
BG, FG, ACCENT, DIM = (18, 24, 34), (230, 235, 240), (94, 177, 255), (140, 150, 160)
GREEN = (126, 211, 133)


def _font(size):
    for name in ("consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_BODY, F_SMALL = _font(25), _font(16), _font(13)


def frame(title, lines, footer="Olist two-stage recommender - capstone demo"):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 22), title[:64], font=F_TITLE, fill=ACCENT)
    d.line([(30, 62), (W - 30, 62)], fill=DIM)
    y = 82
    for line in lines:
        color = FG
        if line.startswith("route:"):
            color = ACCENT
        elif line.startswith("  why:") or line.startswith("       "):
            color = GREEN
        d.text((30, y), line[:100], font=F_BODY, fill=color)
        y += 24
    d.text((30, H - 32), footer, font=F_SMALL, fill=DIM)
    return np.asarray(img)


def table_lines(body, n=8):
    lines = [f"route: {body['route']}", ""]
    lines.append(f"{'#':<3} {'product':<14} {'category':<28} {'BRL':>8}  {'score':>6}")
    for i, it in enumerate(body["items"][:n], 1):
        price = "-" if it["price_brl"] is None else f"{it['price_brl']:.0f}"
        lines.append(f"{i:<3} {it['product_id'][:12] + '..':<14} "
                     f"{str(it['category'])[:26]:<28} {price:>8}  {it['score']:>6}")
    return lines


def explain_lines(body):
    lines = [f"route: {body['route']}", ""]
    for i, it in enumerate(body["items"], 1):
        price = "-" if it["price_brl"] is None else f"{it['price_brl']:.0f}"
        lines.append(f"{i}  {it['product_id'][:12]}..  {str(it['category'])[:26]}"
                     f"  R$ {price}")
        blurb = it.get("explanation") or "(no explanation available)"
        wrapped = textwrap.wrap(blurb, width=88)
        lines.append(f"  why: {wrapped[0]}")
        lines += [f"       {w}" for w in wrapped[1:]]
        lines.append("")
    return lines


def main():
    from app.main import app

    delivered, _ = load_clean_orders()
    counts = delivered.groupby("customer_unique_id")["order_id"].nunique()
    repeat_id = counts[counts >= 2].index[0]

    panels = []  # (seconds, frame)
    with TestClient(app) as client:
        from app.main import STATE  # same cold-customer pick as src.llm_explain.prewarm
        known_users = set(STATE["art"]["im_fw"].user_index)
        cold_id = next(c for c in STATE["art"]["geo"].index if c not in known_users)

        panels.append((3.5, frame("Olist product recommender", [
            "Two-stage pipeline: hybrid candidates + XGBoost re-rank",
            "Cold-start routing: history -> two_stage,",
            "                    known region -> regional_popularity,",
            "                    unknown -> global_popularity",
            "",
            "Step 9: cached LLM explanations (claude-opus-4-8) per recommendation",
            "",
            "GET /health",
            json.dumps(client.get("/health").json(), indent=2),
        ])))

        r = client.get(f"/recommend/{repeat_id}?k=8").json()
        panels.append((6, frame(f"GET /recommend/{repeat_id[:16]}...  (repeat buyer)",
                                table_lines(r))))

        r = client.get(f"/recommend/{cold_id}?k=8").json()
        panels.append((5, frame(f"GET /recommend/{cold_id[:16]}...  (region known, no history)",
                                table_lines(r))))

        r = client.get("/recommend/first-time-visitor?k=8").json()
        panels.append((5, frame("GET /recommend/first-time-visitor  (unknown)",
                                table_lines(r))))

        r = client.get(f"/recommend/{repeat_id}?k=3&explain=true").json()
        panels.append((9, frame("...&explain=true  -  'why you're seeing this' (LLM)",
                                explain_lines(r))))

        r = client.get("/recommend/first-time-visitor?k=3&explain=true").json()
        panels.append((8, frame("cold-start + explain=true  (grounded in SHAP signals)",
                                explain_lines(r))))

        panels.append((4.5, frame("Generative AI in this project (Step 9)", [
            "1. LLM recommendation explanations - the ranker's per-item SHAP",
            "   attributions are glossed into plain English and Claude writes one",
            "   grounded sentence; responses cached in models/explanations_cache.json",
            "",
            "2. LLM-drafted data dictionary - docs/data_dictionary.md",
            "   (prompt, raw draft, and human-verified final all committed)",
            "",
            "Docs: docs/llm/  -  evaluation: notebooks/03-04",
            "",
            "uvicorn app.main:app --port 8000",
        ])))

    out = REPO_ROOT / "docs" / "media"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "demo.mp4"
    with imageio.get_writer(path, fps=FPS, codec="libx264",
                            pixelformat="yuv420p") as writer:
        for seconds, img in panels:
            for _ in range(int(seconds * FPS)):
                writer.append_data(img)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, "
          f"{sum(s for s, _ in panels):.0f}s)")


if __name__ == "__main__":
    main()
