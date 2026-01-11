
import streamlit as st
import backend
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

# -------------------- 1. SETUP --------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="AutoCurve Assistant",
    page_icon="🚘",
    layout="wide"
)

# -------------------- 2. STYLING --------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stVerticalBlock"] > div {
        background-color: #161b22;
        border-radius: 10px;
        border: 1px solid #30363d;
        padding: 20px;
    }
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        padding: 12px;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover { background-color: #2ea043; }
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(45deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    h3 { text-align: center; color: #8b949e; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)



# -------------------- 4. DATA LOADING (HARD FAIL) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "database.xlsx")

st.caption(f"Database path: {DATA_PATH}")
st.caption(f"Files in directory: {os.listdir(BASE_DIR)}")

df = backend.load_data(DATA_PATH)

if df is None or df.empty:
    st.error("Database failed to load or is empty. Fix the file path.")
    st.stop()

required_cols = {"manufacturer", "model", "year", "price", "odometer"}
if not required_cols.issubset(df.columns):
    st.error(f"Database missing required columns: {required_cols - set(df.columns)}")
    st.stop()

all_makes = sorted(df["manufacturer"].dropna().unique())

# -------------------- 5. INPUT UI --------------------
left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Vehicle Images")
    uploaded_files = st.file_uploader(
        "Upload vehicle images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        images = [Image.open(f) for f in uploaded_files]
        cols = st.columns(min(len(images), 3))  # Display up to 3 per row
        for i, img in enumerate(images):
            with cols[i % 3]:
                st.image(img, use_container_width=True)

with right:
    st.subheader("Vehicle Details")

    c1, c2 = st.columns(2)
    with c1:
        selected_make = st.selectbox(
            "Manufacturer",
            all_makes
        )

    with c2:
        subset = df[df["manufacturer"] == selected_make]
        models = sorted(subset["model"].dropna().unique())
        selected_model = st.selectbox("Model", models)

    c3, c4 = st.columns(2)
    with c3:
        year = st.number_input(
            "Year",
            min_value=1990,
            max_value=2026,
            value=2018
        )
    with c4:
        odometer = st.number_input(
            "Odometer (km)",
            min_value=0,
            value=80000,
            step=1000
        )

    # Additional inputs for fuel, transmission, drive
    c5, c6 = st.columns(2)
    with c5:
        fuels = sorted(subset["fuel"].dropna().unique()) if "fuel" in subset else []
        selected_fuel = st.selectbox("Fuel Type", fuels)

    with c6:
        transmissions = sorted(subset["transmission"].dropna().unique()) if "transmission" in subset else []
        selected_transmission = st.selectbox("Transmission Type", transmissions)

    selected_drive = st.selectbox(
        "Drive Type",
        sorted(subset["drive"].dropna().unique()) if "drive" in subset else []
    )

    run_btn = st.button("Run Valuation")

# -------------------- 6. EXECUTION --------------------
if run_btn:
    if not API_KEY:
        st.error("GOOGLE_API_KEY missing in .env")
        st.stop()

    if not uploaded_files:
        st.error("At least one image required for condition analysis.")
        st.stop()

    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        vision = backend.analyze_image_condition(model, images)
        condition_score = vision.get("condition_score", 1.0)

        price, fig, similar, err = backend.run_valuation_model(
            df,
            selected_make,
            selected_model,
            year,
            odometer,
            vision,
            selected_fuel,
            selected_transmission,
            selected_drive
        )

        if err:
            st.error(err)
            st.stop()

        reviews = backend.get_social_proof(
            model,
            f"{year} {selected_make} {selected_model}"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Value", f"${price:,.0f}")
        m2.metric("Condition Multiplier", f"{condition_score:.2f}")
        m3.metric("Comparable Listings", len(similar))

        t1, t2, t3 = st.tabs(["Market Graph", "Image Analysis", "Discussions"])

        with t1:
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                similar.style.format({
                    "price": "${:,.0f}",
                    "odometer": "{:,.0f}"
                })
            )

        with t2:
            l, r = st.columns(2)
            with l:
                for img in images:
                    st.image(img, use_container_width=True)
            with r:
                r.write(vision.get("reasoning", "No explanation provided"))
                for d in vision.get("visible_defects", []):
                    r.error(d)

        with t3:
            for r in reviews:
                st.write(f"**{r['title']}**")
                st.write(r["snippet"])
                st.write(r["link"])

    except Exception as e:
        st.error(f"Runtime error: {e}")