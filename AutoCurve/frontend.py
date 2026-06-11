import streamlit as st
import backend
from PIL import Image
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "google/gemma-3-12b-it:free")

st.set_page_config(
    page_title="AutoCurve — AI Car Valuation",
    page_icon="🚘",
    layout="wide",
)

st.markdown("""
<style>
  /* Base */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { display: none; }

  /* Typography */
  h1, h2, h3, h4 { color: #e6edf3 !important; }
  p, label, .stMarkdown { color: #8b949e; }

  /* Hero */
  .hero { text-align: center; padding: 3rem 1rem 2rem; }
  .hero h1 {
    font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #3fb950, #58a6ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { font-size: 1.15rem; color: #8b949e; max-width: 560px; margin: 0 auto 1.5rem; }

  /* Step badges */
  .step-badge {
    display: inline-block; background: #21262d; border: 1px solid #30363d;
    color: #58a6ff; font-size: 0.72rem; font-weight: 700;
    padding: 2px 10px; border-radius: 20px; letter-spacing: 0.05em;
    text-transform: uppercase; margin-bottom: 8px;
  }

  /* Cards */
  .card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
  }

  /* Metric override */
  [data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 1rem;
  }
  [data-testid="stMetricValue"] { color: #3fb950 !important; font-size: 1.6rem !important; }

  /* Buttons */
  div.stButton > button {
    background: #238636; color: #fff; border: none;
    padding: 0.65rem 1.5rem; font-weight: 600; border-radius: 8px;
    font-size: 1rem; width: 100%; transition: background 0.2s;
  }
  div.stButton > button:hover { background: #2ea043; }

  /* Divider */
  hr { border-color: #21262d; margin: 2rem 0; }

  /* Tabs */
  button[data-baseweb="tab"] { color: #8b949e !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #e6edf3 !important; border-bottom-color: #3fb950 !important; }

  /* Info pill */
  .pill {
    display: inline-block; background: #1f2937;
    border: 1px solid #374151; border-radius: 6px;
    padding: 4px 12px; font-size: 0.8rem; color: #9ca3af;
  }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚘 AutoCurve</h1>
  <p>AI-powered car valuation that combines real market data with visual condition
  scoring to give you a fair, data-driven price estimate.</p>
</div>
""", unsafe_allow_html=True)

# How it works
with st.expander("How does AutoCurve work?", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**① Enter details**\nMake, model, year, mileage, and optional specs.")
    c2.markdown("**② Upload images**\nUp to 4 photos of the vehicle.")
    c3.markdown("**③ Base estimate**\nLinear regression on 70 000+ real listings.")
    c4.markdown("**④ Condition score**\nGemini AI scores visual condition (0.4 – 1.4×) to give the final adjusted price.")

st.markdown("---")

# ── Data ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "database.xlsx")

@st.cache_data(show_spinner="Loading vehicle database…")
def load_df(path):
    return backend.load_data(path)

df = load_df(DATA_PATH)

if df is None or df.empty:
    st.error("Vehicle database failed to load. Check that `database.xlsx` is present.")
    st.stop()

required_cols = {"manufacturer", "model", "year", "price", "odometer"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Database is missing required columns: {missing}")
    st.stop()

all_makes = sorted(df["manufacturer"].dropna().unique())


# ── Step 1 & 2: Inputs ───────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 — Vehicle Details</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        selected_make = st.selectbox("Manufacturer", all_makes, index=all_makes.index("ford") if "ford" in all_makes else 0)
    with r1c2:
        models_for_make = sorted(df[df["manufacturer"] == selected_make]["model"].dropna().unique())
        selected_model = st.selectbox("Model", models_for_make)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        year = st.number_input("Year", min_value=1980, max_value=2026, value=2018, step=1)
    with r2c2:
        odometer = st.number_input("Odometer (miles)", min_value=0, max_value=999_999, value=80_000, step=1_000)

    # Optional filters — scoped to the selected make+model
    def build_options(col):
        subset = df[(df["manufacturer"] == selected_make) & (df["model"] == selected_model)]
        if col not in subset.columns:
            return ["Any"]
        vals = subset[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        vals = sorted(v for v in vals if v and v != "nan")
        return ["Any"] + vals

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        selected_fuel = st.selectbox("Fuel Type", build_options("fuel"))
    with r3c2:
        selected_transmission = st.selectbox("Transmission", build_options("transmission"))

    selected_drive = st.selectbox("Drive Type", build_options("drive"))

with col_right:
    st.markdown('<div class="step-badge">Step 2 — Vehicle Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload 1–4 photos of the vehicle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Clear photos from multiple angles help the AI score condition more accurately.",
    )
    if uploaded_files:
        images_pil = [Image.open(f) for f in uploaded_files[:4]]
        thumb_cols = st.columns(min(len(images_pil), 4))
        for i, img in enumerate(images_pil):
            with thumb_cols[i]:
                st.image(img, use_container_width=True)
        if len(uploaded_files) > 4:
            st.caption("Only the first 4 images are used for analysis.")
    else:
        st.info("Upload at least one photo to enable condition scoring.")

st.markdown("---")

# ── Run ───────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("Estimate Value →", type="primary")

if run_btn:
    # Input validation
    if not API_KEY:
        st.error("**Missing API key.** Set `OPENROUTER_API_KEY` in your `.env` file.")
        st.stop()

    if not uploaded_files:
        st.error("**No images uploaded.** At least one photo is required for condition analysis.")
        st.stop()

    if not selected_make or not selected_model:
        st.error("Please select a make and model.")
        st.stop()

    images_pil = [Image.open(f) for f in uploaded_files[:4]]

    fuel_arg  = None if selected_fuel == "Any" else selected_fuel
    trans_arg = None if selected_transmission == "Any" else selected_transmission
    drive_arg = None if selected_drive == "Any" else selected_drive

    # Step 3: Base estimate
    with st.spinner("Step 3 — Running market regression…"):
        placeholder_vision = {"condition": "good", "condition_score": 1.0}
        base_price, fig, similar, err = backend.run_valuation_model(
            df, selected_make, selected_model, year, odometer,
            placeholder_vision, fuel_arg, trans_arg, drive_arg,
        )

    if err:
        st.error(f"**Valuation error:** {err}")
        st.stop()

    # Step 4: Condition score
    with st.spinner("Step 4 — Scoring visual condition with AI…"):
        client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
        vision = backend.analyze_image_condition(client, MODEL_ID, images_pil)

    condition_score = vision.get("condition_score", 1.0)

    if vision.get("error") == "rate_limited":
        retry = vision.get("retry_after", 60)
        st.warning(f"Rate limit hit — condition score defaulted to 1.0 (neutral). Retry in {retry}s.")
    elif vision.get("error") == "api_failure":
        st.warning("AI condition scoring failed — condition score defaulted to 1.0 (neutral).")

    # Step 5: Final price
    # Re-run with actual vision result so condition_price weighting also uses real condition
    final_price, fig, similar, err = backend.run_valuation_model(
        df, selected_make, selected_model, year, odometer,
        vision, fuel_arg, trans_arg, drive_arg,
    )

    if err:
        st.error(f"**Valuation error:** {err}")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="step-badge">Step 5 — Results</div>', unsafe_allow_html=True)
    st.markdown(f"### Estimated value for your **{year} {selected_make.title()} {selected_model.title()}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Estimated Value", f"${final_price:,.0f}")
    m2.metric("Condition Score", f"{condition_score:.2f}×",
              help="Multiplier applied to base price. Range: 0.4 (salvage) – 1.4 (new).")
    m3.metric("AI Condition", vision.get("condition", "—").title())
    m4.metric("Comparable Listings", len(similar))

    # Explanation
    base_used = final_price / condition_score if condition_score else final_price
    st.markdown(f"""
> **How this estimate was calculated:**
> A weighted regression on **{len(similar)} comparable listings** produced a base price of **${base_used:,.0f}**.
> The AI scored your vehicle's visual condition as **{vision.get('condition', 'good').title()}**
> (score: **{condition_score:.2f}×**), giving a final estimate of
> **${final_price:,.0f}** = ${base_used:,.0f} × {condition_score:.2f}.
""")

    tab1, tab2, tab3 = st.tabs(["📈 Market Graph", "🔍 Image Analysis", "💬 Community Discussions"])

    with tab1:
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Comparable listings used in analysis:**")
        st.dataframe(
            similar.style.format({
                "price": "${:,.0f}",
                "odometer": "{:,.0f}",
            }),
            use_container_width=True,
        )

    with tab2:
        left_img, right_info = st.columns([1, 1])
        with left_img:
            for img in images_pil:
                st.image(img, use_container_width=True)
        with right_info:
            st.markdown(f"**Condition:** {vision.get('condition', '—').title()}")
            st.markdown(f"**Score:** {condition_score:.2f} / 1.4")
            st.progress(min(1.0, (condition_score - 0.4) / 1.0))
            st.markdown("**AI Reasoning:**")
            st.write(vision.get("reasoning", "No explanation provided."))
            defects = vision.get("visible_defects", [])
            if defects:
                st.markdown("**Visible Defects:**")
                for d in defects:
                    st.error(f"⚠ {d}")
            else:
                st.success("No visible defects noted.")

    with tab3:
        with st.spinner("Fetching community discussions…"):
            reviews = backend.get_social_proof(
                f"{year} {selected_make} {selected_model} common problems reliability"
            )
        if reviews:
            for item in reviews:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.write(item["snippet"])
                st.markdown("---")
        else:
            st.info("No discussions found. Try a more specific search.")
