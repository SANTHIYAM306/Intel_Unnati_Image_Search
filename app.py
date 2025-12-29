import streamlit as st
import pandas as pd
import os
import time
from engine import SearchEngine
from PIL import Image

# 1. Advanced Page Config
st.set_page_config(
    page_title="Intel® Unnati | Fashion Neural Search",
    page_icon="🛍️",
    layout="wide"
)

# 2. Senior-Level UI Styling (Glassmorphism & Intel Blue Palette)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #eef2f3, #8e9eab);
    }

    /* Glassmorphism Card Effect */
    .product-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .product-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        background: rgba(255, 255, 255, 0.9);
    }

    .stButton>button {
        background: linear-gradient(135deg, #0071c5 0%, #00c7fd 100%);
        color: white; border: none; border-radius: 12px;
        font-weight: 600; letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }

    .status-box {
        padding: 10px; border-radius: 10px; border-left: 5px solid #0071c5;
        background: white; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Intelligent Resource Management
@st.cache_resource
def load_assets():
    engine = SearchEngine()
    if os.path.exists("styles.csv"):
        df = pd.read_csv("styles.csv", on_bad_lines='skip')
        df['id'] = df['id'].astype(str)
        return engine, df
    return engine, None

engine, styles_df = load_assets()

# 4. Professional Sidebar Logic
with st.sidebar:
    st.image("https://www.intel.com/content/dam/www/central-libraries/us/en/images/intel-logo-blue-background.png", width=120)
    st.markdown("### ⚙️ System Intelligence")
    
    # Auto-Load Logic
    if 'index' not in st.session_state:
        idx, pths = engine.load_db()
        if idx:
            st.session_state.index, st.session_state.paths = idx, pths
            st.success("🧠 Neural Weights Loaded")
    
    if st.button("🔄 Optimize & Re-build Index"):
        with st.status("Vectorizing Dataset...", expanded=True) as s:
            st.session_state.index, st.session_state.paths = engine.create_db("dataset")
            s.update(label="Index Optimization Complete!", state="complete")
            st.balloons()

# 5. Main Content Architecture
st.title("🛍️ AI Fashion Discovery Platform")
st.caption("Industrial-grade visual search powered by Intel® Hardware Acceleration.")

upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload Image to Find Matches", type=['jpg', 'jpeg', 'png'])

# 6. Recommendation Logic (The "Power" Section)
if uploaded_file and 'index' in st.session_state:
    # UX: Visual feedback during processing
    placeholder = st.empty()
    placeholder.info("🔍 Analyzing visual semantic features...")
    
    with open("query.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Identify Category Filtering
    query_id = uploaded_file.name.split('.')[0]
    meta_query = styles_df[styles_df['id'] == query_id] if styles_df is not None else pd.DataFrame()
    target_cat = meta_query.iloc[0]['subCategory'] if not meta_query.empty else None

    # Vector Search (Deep retrieval)
    q_feat = engine.extract("query.jpg").reshape(1, -1).astype('float32')
    # Fetch 200 candidates to ensure high quality after filtering
    scores, idxs = st.session_state.index.search(q_feat, 200)
    
    placeholder.empty()
    
    st.markdown(f"### ✨ Recommendations for you")
    if target_cat:
        st.write(f"Refining search to category: **{target_cat}**")

    # Filter and Collect Results
    matches = []
    for i in range(200):
        if len(matches) >= 20: break # Show top 20 high-quality matches
        
        p = st.session_state.paths[idxs[0][i]]
        m_id = os.path.basename(p).split('.')[0]
        m_info = styles_df[styles_df['id'] == m_id] if styles_df is not None else pd.DataFrame()
        
        # Logic: Category must match if available
        if not m_info.empty:
            if target_cat and m_info.iloc[0]['subCategory'] != target_cat:
                continue
            matches.append((p, m_info.iloc[0], scores[0][i]))

    # Grid Display (Clean 4-column layout)
    if matches:
        rows = [matches[i:i+4] for i in range(0, len(matches), 4)]
        for row in rows:
            cols = st.columns(4)
            for j, (img_p, meta, score) in enumerate(row):
                with cols[j]:
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    st.image(img_p, use_container_width=True)
                    st.markdown(f"**{meta['productDisplayName'][:22]}...**")
                    
                    # Pro Metric: Confidence bar
                    s_val = float(score)
                    st.progress(max(0.0, min(1.0, s_val)))
                    st.caption(f"Match Confidence: {s_val*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No high-confidence matches found in this category.")

else:
    st.write("---")
    st.markdown("#### 👋 Get Started\n1. Ensure your `dataset/images` folder is populated.\n2. Click **Re-build Index** in the sidebar.\n3. Upload a fashion photo to see the magic.")