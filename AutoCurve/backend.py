import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from openai import OpenAI
import json
import base64
import io
from  duckduckgo_search import DDGS
import time
from rate_limiter import OPENROUTER_LIMITER


def load_data(path):
    try:
        df = pd.read_excel(path)
        # Normalize string columns for consistency
        cols_to_normalize = ["manufacturer", "model", "condition", "fuel", "transmission", "drive", "title_status", "state"]
        for col in cols_to_normalize:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def analyze_image_condition(client, model_id, images):

    if not OPENROUTER_LIMITER.allow(1):
        wait = OPENROUTER_LIMITER.retry_after_seconds(1)
        return {
            "ok": False,
            "error": "rate_limited",
            "retry_after": round(wait, 2)
        }
    prompt = """
    Analyze these vehicle images for its overall condition. Classify the condition as one of:
    new, like new, excellent, good, fair, salvage.


    Provide reasoning, a small list of visible defects(weather conditions are not defects i.e rain,snow), and a condition score from 0 to 1.4
    where average is the mean for the 2 values for the classified condition.

    Output ONLY valid JSON with keys:
    condition, reasoning, visible_defects (list), condition_score (number).
    """

    # Convert PIL images -> data URLs for OpenRouter
    parts = [{"type": "text", "text": prompt}]
    for img in images[:4]:  # 1-4 images
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # 2) OpenRouter call with retry/backoff on rate limit
    last_err = None
    resp = None

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": parts}],
                temperature=0.2,
                max_tokens=400
            )
            last_err = None
            break

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Only retry if it's likely a rate limit
            if "429" in msg or "rate limit" in msg or "too many" in msg:
                sleep_time = min(30, 2 ** attempt)  # add jitter if you want
                time.sleep(sleep_time)
                continue

            # Not a rate limit error → crash normally
            raise

    if last_err is not None or resp is None:
        return {
            "condition": "good",
            "reasoning": f"OpenRouter rate limited repeatedly: {last_err}",
            "visible_defects": [],
            "condition_score": 1.0
        }

    # 3) Parse model output
    vision_text = resp.choices[0].message.content or ""
    vision_text = vision_text.replace("```json", "").replace("```", "").strip()

    try:
        vision = json.loads(vision_text)
    except json.JSONDecodeError:
        vision = {
            "condition": "good",
            "reasoning": f"Failed to parse model output. Raw: {vision_text[:300]}",
            "visible_defects": [],
            "condition_score": 1.0
        }

    return vision

def run_valuation_model(df, make, model, year, odometer, vision, fuel=None, transmission=None, drive=None):
    make = str(make).strip().lower()
    model = str(model).strip().lower()
    condition_str = vision.get("condition", "good")
    condition_score = vision.get("condition_score", 1.0)

    # Base filter: same make and model
    filter_base = df[(df['manufacturer'] == make) & (df['model'] == model)]
    if filter_base.empty:
        return None, None, None, "No data found for the selected make and model."

    # Regressions for year and odometer 
    if len(filter_base) < 2:
        year_price = filter_base['price'].mean()
        odo_price = year_price
    else:
        reg_year = LinearRegression().fit(filter_base[['year']], filter_base['price'])
        year_price = reg_year.predict(pd.DataFrame({'year': [year]}))[0]

        reg_odo = LinearRegression().fit(filter_base[['odometer']], filter_base['price'])
        odo_price = reg_odo.predict(pd.DataFrame({'odometer': [odometer]}))[0]

    # Categorical filter for P (manufacturer, model, fuel, transmission, drive)
    filter_cat = filter_base.copy()
    if fuel:
        filter_cat = filter_cat[filter_cat['fuel'] == str(fuel).strip().lower()]
    if transmission:
        filter_cat = filter_cat[filter_cat['transmission'] == str(transmission).strip().lower()]
    if drive:
        filter_cat = filter_cat[filter_cat['drive'] == str(drive).strip().lower()]
    if filter_cat.empty:
        filter_cat = filter_base  # Relax filters if no matches

    # Get 2 lower and 2 higher odometer cars for cat_price (2x2 matrix equivalent)
    lower_cat = filter_cat[filter_cat['odometer'] < odometer].sort_values('odometer', ascending=False).head(2)
    higher_cat = filter_cat[filter_cat['odometer'] > odometer].sort_values('odometer').head(2)
    closest_cat = pd.concat([lower_cat, higher_cat])
    cat_price = closest_cat['price'].mean() if not closest_cat.empty else filter_base['price'].mean()

    # Condition filter for condition_price
    filter_cond = filter_base[filter_base['condition'] == condition_str]
    if filter_cond.empty:
        condition_price = cat_price 
    else:
        lower_cond = filter_cond[filter_cond['odometer'] < odometer].sort_values('odometer', ascending=False).head(2)
        higher_cond = filter_cond[filter_cond['odometer'] > odometer].sort_values('odometer').head(2)
        closest_cond = pd.concat([lower_cond, higher_cond])
        condition_price = closest_cond['price'].mean() if not closest_cond.empty else filter_base['price'].mean()

    # Weighted sum
    base_price = (0.15 * cat_price) + (0.250 * condition_price) + (0.30 * year_price) + (0.35 * odo_price)

    # Final price adjusted by condition score from image
    price = base_price * condition_score

    # Plot: Market graph of price vs odometer with trendline
    fig = px.scatter(filter_base, x='odometer', y='price', trendline="ols", title="Market Prices for Similar Vehicles")

    similar = filter_cat if not filter_cat.empty else filter_base
    similar = similar[['year', 'model', 'odometer', 'price', 'condition', 'fuel', 'transmission', 'drive']].head(10)  # Limit for display

    return price, fig, similar, None

def get_social_proof(query: str, max_results: int = 3):
    q = f'{query} site:reddit.com'
    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(q, max_results=max_results):
            out.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "link": r.get("href", "")
            })
    return out