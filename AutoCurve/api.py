"""
AutoCurve FastAPI backend.
Run: uvicorn api:app --reload
"""
import io
import os
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

import backend

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY  = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "google/gemma-3-12b-it:free")

app = FastAPI(title="AutoCurve", docs_url=None, redoc_url=None)

# Load dataset once at startup
_df = backend.load_data(os.path.join(BASE_DIR, "database.xlsx"))


def _require_df():
    if _df is None or _df.empty:
        raise HTTPException(500, "Vehicle database failed to load.")
    return _df


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.get("/api/makes")
def get_makes():
    df = _require_df()
    return {"makes": sorted(df["manufacturer"].dropna().unique().tolist())}


@app.get("/api/models")
def get_models(make: str):
    df = _require_df()
    sub = df[df["manufacturer"] == make.strip().lower()]
    return {"models": sorted(sub["model"].dropna().unique().tolist())}


@app.get("/api/options")
def get_options(make: str, model: str):
    df = _require_df()
    sub = df[
        (df["manufacturer"] == make.strip().lower()) &
        (df["model"]        == model.strip().lower())
    ]

    def vals(col):
        if col not in sub.columns:
            return []
        v = sub[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        return sorted(x for x in v if x and x != "nan")

    return {
        "fuel":         vals("fuel"),
        "transmission": vals("transmission"),
        "drive":        vals("drive"),
    }


# ── Valuation endpoint ────────────────────────────────────────────────────────

@app.post("/api/valuation")
async def valuation(
    make:         str           = Form(...),
    model:        str           = Form(...),
    year:         int           = Form(...),
    odometer:     int           = Form(...),
    fuel:         Optional[str] = Form(None),
    transmission: Optional[str] = Form(None),
    drive:        Optional[str] = Form(None),
    images: List[UploadFile]    = File(...),
):
    df = _require_df()

    if not API_KEY:
        raise HTTPException(400, "OPENROUTER_API_KEY is not set. Add it to your .env file.")
    if not images:
        raise HTTPException(400, "At least one image is required.")
    if not (1980 <= year <= 2026):
        raise HTTPException(400, "Year must be between 1980 and 2026.")
    if not (0 <= odometer < 1_000_000):
        raise HTTPException(400, "Odometer must be between 0 and 999,999 miles.")

    # Parse uploaded images
    pil_images: List[Image.Image] = []
    for f in images[:4]:
        raw = await f.read()
        try:
            pil_images.append(Image.open(io.BytesIO(raw)))
        except Exception:
            raise HTTPException(400, f"Could not read image '{f.filename}'. Upload a valid JPG or PNG.")

    # AI condition scoring
    client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
    vision = backend.analyze_image_condition(client, MODEL_ID, pil_images)

    # Normalise optional filters
    fuel_arg  = fuel         if fuel         and fuel.lower()         not in ("", "any") else None
    trans_arg = transmission if transmission and transmission.lower() not in ("", "any") else None
    drive_arg = drive        if drive        and drive.lower()        not in ("", "any") else None

    final_price, _fig, similar, err = backend.run_valuation_model(
        df, make, model, year, odometer, vision, fuel_arg, trans_arg, drive_arg
    )
    if err:
        raise HTTPException(400, err)

    condition_score = vision.get("condition_score", 1.0)
    base_price      = final_price / condition_score if condition_score else final_price

    # Market scatter data for the chart (capped at 500 points)
    filt = df[
        (df["manufacturer"] == make.strip().lower()) &
        (df["model"]        == model.strip().lower())
    ]
    market_pts = filt[["odometer", "price"]].dropna()
    if len(market_pts) > 500:
        market_pts = market_pts.sample(500, random_state=42)
    market_data = market_pts.to_dict(orient="records")

    return {
        "final_price":         round(final_price),
        "base_price":          round(base_price),
        "condition":           vision.get("condition", "good"),
        "condition_score":     condition_score,
        "reasoning":           vision.get("reasoning", ""),
        "visible_defects":     vision.get("visible_defects", []),
        "comparable_count":    len(similar),
        "comparable_listings": similar.fillna("").to_dict(orient="records"),
        "market_data":         market_data,
        "api_error":           vision.get("error"),
        "retry_after":         vision.get("retry_after"),
    }


# ── Static files ──────────────────────────────────────────────────────────────

static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))
