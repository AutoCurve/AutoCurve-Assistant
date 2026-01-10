import streamlit as st
#import backend
from PIL import Image
import time

# --- 1. CONFIGURATION & CUSTOM CSS ---
st.title("AutoCurve Assistant") 
st.set_page_config(
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)
#manufac,model,fuel type, odometer,title,transmission,year,
st.markdown("""
<style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark Theme Colors */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Container Styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    /* Metrics Numbers */
    div[data-testid="stMetricValue"] {
        font-size: 36px;
        color: #4CAF50;
        font-weight: 700;
    }
    
    /* Green Action Button */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }
    
    /* Hero Title Gradient */
    .hero-title {
        font-size: 60px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 10px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d;
        border-radius: 4px;
        padding: 10px 20px;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HEADER SECTION ---
st.markdown("<h3 style='text-align: center; color: #8b949e; margin-bottom: 40px;'>The AI-Powered Vehicle Appraiser</h3>", unsafe_allow_html=True)


# --- 3. INPUT SECTION (User View) ---
# Centered layout for the uploader
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    uploaded_file = st.file_uploader("Upload Vehicle Photo", type=["jpg", "png", "jpeg"])
#manufac,model,fuel type, odometer,title,transmission,year,

manufacturers = ["acura","alfa-romeo","aston-martin","audi","bmw",
                 "buick","cadillac","chevrolet","chrysler","dodge","ferrari","fiat","ford",
                 "gmc","harley-davidson","honda","hyundai","infiniti","jeep","kia","lexus",
                 "lincoln","mazda","mercedes-benz","mercury","mini","mitsubishi","nissan",
                 "pontiac","porsche","ram","rover","saturn","subaru","tesla","toyota","volkswagen","volvo"]
year = st.number_input("Enter the year (1999-2021)", min_value = 1999, max_value= 2021, step = 1, format="%d")
st.selectbox("Manufacturers", manufacturers)


# manufactor = st.selectmanufacturers = sorted(df["manufacturer"].dropna().unique())
# st.selectbox("Manufacturer", manufacturers)box("")



# --- 4. MAIN APP LOGIC ---
if uploaded_file:
    st.markdown("---")
    
    # Split Layout: Image (Left) | Analysis (Right)
    img_col, info_col = st.columns([1, 1], gap="large")
    
    with img_col:
        image = Image.open(uploaded_file)
        st.image(image, caption="Vehicle Preview", use_container_width=True)
        
        # The Big Green Button
        analyze_btn = st.button("✨ GENERATE VALUATION REPORT")

    if analyze_btn:
        with info_col:
            # Progress Container
            status_container = st.container()
            with status_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # --- STEP 1: VISION SCAN ---
                status_text.markdown(" 👁️ Scanning image for defects...")
                progress_bar.progress(10)
                
                # Call Backend (No API key passed - backend handles it)
                try:
                    scan_data = backend.identify_and_scan(image)
                    car_name = scan_data.get("identity", "Unknown Car")
                    
                    # Update Progress
                    status_text.markdown(f"### 📂 Loading market data for **{car_name}**...")
                    progress_bar.progress(40)
                    
                    # --- STEP 2: LOAD DATA ---
                    df = backend.load_market_data()
                    filtered_df = backend.filter_similar_cars(df, car_name)
                    market_csv = filtered_df[['year', 'model', 'price', 'condition']].to_csv(index=False)
                    
                    # Update Progress
                    status_text.markdown("### 🌐 Checking Reddit for reliability...")
                    progress_bar.progress(70)
                    
                    # --- STEP 3: SOCIAL PROOF ---
                    reddit_txt, reddit_links = backend.get_social_proof(car_name)
                    
                    # Update Progress
                    status_text.markdown("### 💰 Finalizing valuation...")
                    progress_bar.progress(90)
                    
                    # --- STEP 4: FINAL CALCULATION ---
                    final_data = backend.get_final_valuation(image, car_name, market_csv, reddit_txt)
                    
                    # Finish
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    # Clear progress bar to show results
                    status_container.empty()
                    
                    # --- DISPLAY RESULTS ---
                    st.success("Analysis Complete")
                    
                    # TABS UI
                    tab1, tab2, tab3 = st.tabs(["💰 VALUATION", "🔍 DEFECTS", "🗣️ REDDIT"])
                    
                    with tab1:
                        st.subheader("Market Assessment")
                        m1, m2 = st.columns(2)
                        m1.metric("Fair Market Range", final_data.get('price_range', "N/A"))
                        m2.metric("Value Score", final_data.get('score', "N/A"))
                        
                        st.markdown("#### 📝 Verdict")
                        st.info(final_data.get('verdict', "No verdict generated."))
                        st.caption(f"Compared against {len(filtered_df)} similar listings in database.")

                    with tab2:
                        st.subheader("Visual Inspection Report")
                        defects = scan_data.get('defects', [])
                        
                        # Logic to show Green if clean, Red if dirty
                        if defects and isinstance(defects, list) and "No major" not in defects[0]:
                            for d in defects:
                                st.error(f"⚠️ {d}")
                        else:
                            st.success("✅ No major visible body defects detected.")

                    with tab3:
                        st.subheader("Owner Sentiment Analysis")
                        st.markdown(reddit_txt)
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        for title, link in reddit_links:
                            st.markdown(f"🔗 [{title}]({link})")
                            
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.error("Check your backend.py API Key configuration.")