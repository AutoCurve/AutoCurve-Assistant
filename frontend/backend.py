import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from openai import OpenAI
import json
import base64
import io

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
    prompt = """
    Analyze these vehicle images for its overall condition. Classify the condition as one of:
    new, like new, excellent, good, fair, salvage.

    Provide reasoning, a list of visible defects, and a condition score from 0 to 1.4
    where average is the mean for the 2 values for the classified condition.

    Output ONLY valid JSON with keys:
    condition, reasoning, visible_defects (list), condition_score (number).
    """

    # Convert PIL images -> data URLs for OpenRouter
    parts = [{"type": "text", "text": prompt}]

    for img in images[:4]:  # don't spam; 1-4 images is enough
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": parts}],
        temperature=0.2,
        max_tokens=400
    )

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

    # Regressions for year and odometer (univariate on base filter)
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
        condition_price = cat_price  # Fallback
    else:
        lower_cond = filter_cond[filter_cond['odometer'] < odometer].sort_values('odometer', ascending=False).head(2)
        higher_cond = filter_cond[filter_cond['odometer'] > odometer].sort_values('odometer').head(2)
        closest_cond = pd.concat([lower_cond, higher_cond])
        condition_price = closest_cond['price'].mean() if not closest_cond.empty else filter_base['price'].mean()

    # Weighted sum: a1=0.25, a2=0.25, a3=0.25, a4=0.25 (adjust if needed)
    base_price = (0.25 * cat_price) + (0.25 * condition_price) + (0.25 * year_price) + (0.25 * odo_price)

    # Final price adjusted by condition score from image
    price = base_price * condition_score

    # Plot: Market graph of price vs odometer with trendline
    fig = px.scatter(filter_base, x='odometer', y='price', trendline="ols", title="Market Prices for Similar Vehicles")

    # Similar listings: Use the categorical closest cars, fallback to base
    similar = filter_cat if not filter_cat.empty else filter_base
    similar = similar[['year', 'model', 'odometer', 'price', 'condition', 'fuel', 'transmission', 'drive']].head(10)  # Limit for display

    return price, fig, similar, None

def get_social_proof(client, model_id, query):
    prompt = f"""
    Generate 3 realistic forum-style discussions about '{query}'.
    Include mixed sentiments.
    Output ONLY valid JSON list of objects with: title, snippet, link.
    """

    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=350
    )

    reviews_text = resp.choices[0].message.content or ""
    reviews_text = reviews_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(reviews_text)
    except json.JSONDecodeError:
        return [
            {"title": "Default Review 1", "snippet": "Great car!", "link": "https://example.com/review1"},
            {"title": "Default Review 2", "snippet": "Average performance.", "link": "https://example.com/review2"},
            {"title": "Default Review 3", "snippet": "Some issues noted.", "link": "https://example.com/review3"},
        ]
