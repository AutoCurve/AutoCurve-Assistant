# AutoCurve — AI-Powered Car Valuation

AutoCurve estimates the fair market value of a used vehicle by combining **regression on 40000+ real listings** with **AI-based visual condition scoring** from uploaded photos.

---

## Problem It Solves

Used-car buyers and sellers struggle to price vehicles accurately. Published guides ignore individual vehicle condition; dealers have information advantages. AutoCurve closes that gap by blending market data with objective, image-based condition assessment — giving anyone a data-driven, photo-backed price estimate in seconds.

---

## Features

- Market regression across 70 000+ Craigslist USA listings
- Per-make/model linear regression on year and odometer
- Optional filtering by fuel type, transmission, and drive
- AI visual condition scoring via OpenRouter (Gemma 4 multimodal LLM) — multiplier clamped to 0.4–1.4×
- Step-by-step UI with market scatter plot, comparable listings, and image analysis breakdown
- Community discussion search (Reddit) via DuckDuckGo
- API rate limiting (token bucket, 15 req/min)
- Full input validation and graceful API-failure handling

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML + vanilla JS (Spellbook-inspired design system) |
| Backend / API | FastAPI + Uvicorn |
| ML | Python, scikit-learn, pandas |
| AI Vision | OpenRouter API (Gemma 4 multimodal LLM) |
| Data | Craigslist used-car dataset (Excel) |
| Testing | pytest |

---

## Architecture / Workflow

```
User Input (make, model, year, odometer, images)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Step 3 — Base Price Estimation                  │
│  • Filter dataset to matching make + model       │
│  • Linear Regression on year  → year_price       │
│  • Linear Regression on odo   → odo_price        │
│  • Nearest-neighbour cat match → cat_price       │
│  • Condition-category match   → condition_price  │
│  • Weighted average → base_price                 │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Step 4 — Visual Condition Scoring               │
│  • Upload images → OpenRouter multimodal API     │
│  • Returns condition label + score [0.4, 1.4]    │
│  • Score clamped server-side regardless of API   │
└──────────────────────────────────────────────────┘
        │
        ▼
  final_price = base_price × condition_score
```

---

## ML Approach

The valuation model uses per-query regression rather than a single global model:

1. Filter listings to the exact make + model.
2. Fit `LinearRegression(year → price)` and `LinearRegression(odometer → price)` on that subset.
3. Find nearest listings by odometer matching the optional categorical filters (fuel, transmission, drive) and by reported condition.
4. Combine with weights: **35% odometer regression, 30% year regression, 25% condition-category match, 15% categorical match**.
5. Multiply by the AI condition score.

For portfolio/resume evaluation a global Linear Regression (with one-hot encoding) is trained separately — see [Model Metrics](#model-metrics).

---

## Image Condition Scoring

Uploaded photos (up to 4) are sent to a multimodal LLM via OpenRouter. The model classifies the vehicle as one of: `new`, `like new`, `excellent`, `good`, `fair`, `salvage`, and returns a `condition_score` in [0.4, 1.4]:

| Condition | Score range |
|---|---|
| new | 1.3 – 1.4 |
| like new | 1.2 – 1.3 |
| excellent | 1.1 – 1.2 |
| good | 0.9 – 1.0 |
| fair | 0.7 – 0.8 |
| salvage | 0.4 – 0.6 |

The score is **always clamped** to [0.4, 1.4] in `backend.py` regardless of what the API returns. On API failure the score defaults to 1.0 (neutral) and the user is warned.

---

## Model Metrics

Evaluated on a global Linear Regression (80/20 split, 52 991 cleaned records):

| Metric | Value |
|---|---|
| MAE | **$3,077** |
| RMSE | **$5,198** |
| R² | **0.84** |
| Train size | 42,392 records (80%) |
| Test size | 10,599 records (20%) |

See [`MODEL_METRICS.md`](MODEL_METRICS.md) for full detail and [`evaluate_model.py`](evaluate_model.py) to reproduce.

---

## Setup

### Prerequisites
- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

### Install

```bash
git clone <repo-url>
cd AutoCurve
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | Your OpenRouter API key |
| `OPENROUTER_MODEL_ID` | ❌ | Vision model to use (default: `google/gemma-4-26b-a4b-it:free`) |

---

## Running the App

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Running Tests

```bash
pytest tests/ -v
```

24 tests covering: condition score clamping, valuation model inputs/outputs, API failure handling, data loading and cleaning.

---

## Reproducing Model Metrics

```bash
python evaluate_model.py
```

This re-trains and re-evaluates the global model and overwrites `MODEL_METRICS.md`.

---

## Deployment

The app is a standard FastAPI service, deployable to any platform that runs Python web servers (Railway, Render, Fly.io).

### Railway (recommended — always-on)

1. Push the repo to GitHub (ensure `.env` is in `.gitignore`).
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**.
3. Set **Root Directory** to `AutoCurve`.
4. Set the **Start Command**:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables in the dashboard:
   - `OPENROUTER_API_KEY` = your key
   - `OPENROUTER_MODEL_ID` = `google/gemma-4-26b-a4b-it:free`
6. Deploy.

### Render

Same as above, but note the free tier spins down after 15 min of inactivity (slow cold start on the next request). Use a paid instance or Railway to keep it always-on.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Important before deploying
- Confirm `database.xlsx` is committed (it is not a secret).
- Confirm `.env` is in `.gitignore` ✅ — secrets go in the platform dashboard only.

---

## Limitations

- Dataset is USA Craigslist listings only; international pricing not covered.
- Listing prices are asking prices, not final sale prices.
- Sparse data for rare makes/models degrades estimates.
- AI condition scoring accuracy depends on image quality and lighting.
- Rate limiter (15 req/min) is in-process — not shared across multiple server instances.

---

## Future Improvements

- VIN lookup for trim/options/accident history
- Regional price adjustment by ZIP code
- Gradient-boosted tree model (XGBoost) for non-linear feature interactions
- Verified sale prices instead of listing prices
- Redis-backed distributed rate limiter for multi-instance deployments
