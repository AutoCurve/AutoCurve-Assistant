import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from openai import OpenAI
import json
import base64
import io
from duckduckgo_search import DDGS
import time
from rate_limiter import OPENROUTER_LIMITER

CONDITION_SCORE_MIN = 0.4
CONDITION_SCORE_MAX = 1.4
PRICE_MIN = 500
PRICE_MAX = 200_000


def load_data(path):
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    cols_to_normalize = ["manufacturer", "model", "condition", "fuel", "transmission", "drive", "title_status", "state"]
    for col in cols_to_normalize:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    #Drop rows with missing critical fields or unrealistic prices
    df = df.dropna(subset=["manufacturer", "model", "year", "price", "odometer"])
    # Remove any rows where normalised string values became "nan"
    for col in cols_to_normalize:
        if col in df.columns:
            df = df[df[col] != "nan"]
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    df = df[(df["odometer"] >= 0) & (df["odometer"] < 1_000_000)]
    df = df[(df["year"] >= 1980) & (df["year"] <= 2026)]

    return df.reset_index(drop=True)


def clamp_condition_score(score):
    """Clamp condition score to the valid [0.4, 1.4] range."""
    try:
        return float(max(CONDITION_SCORE_MIN, min(CONDITION_SCORE_MAX, float(score))))
    except (TypeError, ValueError):
        return 1.0


def analyze_image_condition(client, model_id, images):
    if not images:
        return {
            "ok": False,
            "error": "no_images",
            "condition": "good",
            "reasoning": "No images provided.",
            "visible_defects": [],
            "condition_score": 1.0,
        }

    if not OPENROUTER_LIMITER.allow(1):
        wait = OPENROUTER_LIMITER.retry_after_seconds(1)
        return {
            "ok": False,
            "error": "rate_limited",
            "retry_after": round(wait, 2),
            "condition": "good",
            "reasoning": "Rate limit reached. Try again shortly.",
            "visible_defects": [],
            "condition_score": 1.0,
        }

    prompt = (
        "Analyze these vehicle images for overall condition. "
        "Classify the condition as one of: new, like new, excellent, good, fair, salvage.\n\n"
        "Provide reasoning, a list of visible defects (weather is not a defect), "
        "and a condition_score between 0.4 and 1.4 where:\n"
        "  new=1.3-1.4, like new=1.2-1.3, excellent=1.1-1.2, "
        "good=0.9-1.0, fair=0.7-0.8, salvage=0.4-0.6\n\n"
        "Output ONLY valid JSON with keys: "
        "condition (string), reasoning (string), visible_defects (list of strings), condition_score (number)."
    )

    parts = [{"type": "text", "text": prompt}]
    for img in images[:4]:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    last_err = None
    resp = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": parts}],
                temperature=0.2,
                max_tokens=400,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "too many" in msg:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise

    if last_err is not None or resp is None:
        return {
            "ok": False,
            "error": "api_failure",
            "condition": "good",
            "reasoning": f"API call failed after retries: {last_err}",
            "visible_defects": [],
            "condition_score": 1.0,
        }

    vision_text = (resp.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()

    try:
        vision = json.loads(vision_text)
    except json.JSONDecodeError:
        vision = {
            "condition": "good",
            "reasoning": f"Could not parse model response. Raw output: {vision_text[:300]}",
            "visible_defects": [],
            "condition_score": 1.0,
        }

    # Always clamp the score regardless of what the model returns
    vision["condition_score"] = clamp_condition_score(vision.get("condition_score", 1.0))
    return vision


def run_valuation_model(df, make, model, year, odometer, vision, fuel=None, transmission=None, drive=None):
    make = str(make).strip().lower()
    model = str(model).strip().lower()
    condition_str = str(vision.get("condition", "good")).strip().lower()
    condition_score = clamp_condition_score(vision.get("condition_score", 1.0))

    filter_base = df[(df["manufacturer"] == make) & (df["model"] == model)]
    if filter_base.empty:
        return None, None, None, f"No listings found for {make.title()} {model.title()} in the dataset."

    if len(filter_base) < 2:
        year_price = filter_base["price"].mean()
        odo_price = year_price
    else:
        reg_year = LinearRegression().fit(filter_base[["year"]], filter_base["price"])
        year_price = float(reg_year.predict(pd.DataFrame({"year": [year]}))[0])

        reg_odo = LinearRegression().fit(filter_base[["odometer"]], filter_base["price"])
        odo_price = float(reg_odo.predict(pd.DataFrame({"odometer": [odometer]}))[0])

    filter_cat = filter_base.copy()
    if fuel:
        fc = filter_cat[filter_cat["fuel"] == str(fuel).strip().lower()]
        if not fc.empty:
            filter_cat = fc
    if transmission:
        fc = filter_cat[filter_cat["transmission"] == str(transmission).strip().lower()]
        if not fc.empty:
            filter_cat = fc
    if drive:
        fc = filter_cat[filter_cat["drive"] == str(drive).strip().lower()]
        if not fc.empty:
            filter_cat = fc

    lower_cat = filter_cat[filter_cat["odometer"] < odometer].sort_values("odometer", ascending=False).head(2)
    higher_cat = filter_cat[filter_cat["odometer"] > odometer].sort_values("odometer").head(2)
    closest_cat = pd.concat([lower_cat, higher_cat])
    cat_price = float(closest_cat["price"].mean()) if not closest_cat.empty else float(filter_base["price"].mean())

    filter_cond = filter_base[filter_base["condition"] == condition_str]
    if filter_cond.empty:
        condition_price = cat_price
    else:
        lower_cond = filter_cond[filter_cond["odometer"] < odometer].sort_values("odometer", ascending=False).head(2)
        higher_cond = filter_cond[filter_cond["odometer"] > odometer].sort_values("odometer").head(2)
        closest_cond = pd.concat([lower_cond, higher_cond])
        condition_price = float(closest_cond["price"].mean()) if not closest_cond.empty else float(filter_base["price"].mean())

    base_price = (0.15 * cat_price) + (0.25 * condition_price) + (0.30 * year_price) + (0.35 * odo_price)
    # Clamp base_price to a sane range before applying condition multiplier
    base_price = max(PRICE_MIN, min(base_price, PRICE_MAX * 2))

    final_price = base_price * condition_score

    fig = px.scatter(
        filter_base,
        x="odometer",
        y="price",
        trendline="ols",
        title=f"Market Prices: {make.title()} {model.title()}",
        labels={"odometer": "Odometer (miles)", "price": "Price (USD)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa",
    )

    display_cols = [c for c in ["year", "model", "odometer", "price", "condition", "fuel", "transmission", "drive"] if c in filter_cat.columns]
    similar = filter_cat[display_cols].head(10)

    return final_price, fig, similar, None


def get_social_proof(query: str, max_results: int = 3):
    try:
        q = f"{query} site:reddit.com"
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=max_results):
                out.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "link": r.get("href", ""),
                })
        return out
    except Exception:
        return []
