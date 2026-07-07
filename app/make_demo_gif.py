"""Generate docs/media/demo.gif from live API responses (capstone Step 8).

Drives the FastAPI app in-process (TestClient), captures /health and two
/recommend calls (a repeat buyer -> personalised route; an unknown visitor ->
cold-start route), and renders the actual JSON into GIF frames. Reproducible
demo media: every number shown comes from a real request.

    python -m app.make_demo_gif
"""

import json

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from src.data import REPO_ROOT, load_table
from src.features import load_clean_orders

W, H = 900, 560
BG, FG, ACCENT, DIM = (18, 24, 34), (230, 235, 240), (94, 177, 255), (140, 150, 160)


def _font(size):
    for name in ("consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_BODY, F_SMALL = _font(26), _font(17), _font(14)


def frame(title, lines, accent_lines=()):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 24), title, font=F_TITLE, fill=ACCENT)
    d.line([(30, 66), (W - 30, 66)], fill=DIM)
    y = 88
    for line in lines:
        color = ACCENT if line in accent_lines else FG
        d.text((30, y), line[:96], font=F_BODY, fill=color)
        y += 26
    d.text((30, H - 34), "Olist two-stage recommender - capstone Step 8 demo",
           font=F_SMALL, fill=DIM)
    return img


def response_lines(body):
    lines = [f"route: {body['route']}", ""]
    lines.append(f"{'#':<3} {'product':<14} {'category':<28} {'BRL':>8}  {'score':>6}")
    for i, it in enumerate(body["items"], 1):
        price = "-" if it["price_brl"] is None else f"{it['price_brl']:.0f}"
        lines.append(f"{i:<3} {it['product_id'][:12] + '..':<14} "
                     f"{str(it['category'])[:26]:<28} {price:>8}  {it['score']:>6}")
    return lines


def main():
    from app.main import app

    # a real repeat buyer makes the personalised route visible in the demo
    delivered, _ = load_clean_orders()
    counts = delivered.groupby("customer_unique_id")["order_id"].nunique()
    repeat_id = counts[counts >= 2].index[0]

    frames = []
    with TestClient(app) as client:
        frames.append(frame("Olist product recommender", [
            "Two-stage pipeline: hybrid candidates + XGBoost re-rank",
            "Cold-start routing: history -> hybrid, cold -> regional popularity",
            "",
            "GET /health",
            json.dumps(client.get("/health").json(), indent=2),
        ]))

        r = client.get(f"/recommend/{repeat_id}?k=8").json()
        frames.append(frame(f"GET /recommend/{repeat_id[:16]}...  (repeat buyer)",
                            response_lines(r), accent_lines=(f"route: {r['route']}",)))

        r2 = client.get("/recommend/first-time-visitor?k=8").json()
        frames.append(frame("GET /recommend/first-time-visitor  (no history)",
                            response_lines(r2), accent_lines=(f"route: {r2['route']}",)))

        frames.append(frame("That's the whole loop", [
            "docs/deployment_guide.md covers Docker, monitoring, rollback",
            "notebooks/03-04 hold the evaluation behind these scores",
            "",
            "uvicorn app.main:app --port 8000",
        ]))

    out = REPO_ROOT / "docs" / "media"
    out.mkdir(parents=True, exist_ok=True)
    frames[0].save(out / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=[2600, 4200, 4200, 3000], loop=0)
    print(f"wrote {out / 'demo.gif'} ({(out / 'demo.gif').stat().st_size / 1e3:.0f} KB)")

    _ = load_table  # keep import referenced


if __name__ == "__main__":
    main()
