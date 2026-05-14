import streamlit as st
import torch
import clip
import numpy as np
import joblib
import json
import requests
from PIL import Image
from io import BytesIO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veritas — News Verification Terminal",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* Terminal header */
.terminal-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

/* Title */
.main-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 4px;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.main-subtitle {
    font-size: 0.85rem;
    color: #555;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Divider */
.terminal-divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}

/* Input panel */
.input-panel {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.panel-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Streamlit input overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #0a0a0f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 6px !important;
    color: #e0e0e0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #4a4aff !important;
    box-shadow: 0 0 0 1px #4a4aff !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f0f1a;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #444;
    padding: 0.6rem 1.5rem;
    border: none;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #4a4aff !important;
    background: transparent !important;
}

/* Button */
.stButton > button {
    background: #4a4aff;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #6060ff;
    transform: translateY(-1px);
}

/* Verdict cards */
.verdict-authentic {
    background: linear-gradient(135deg, #001a0e, #002a16);
    border: 1px solid #00ff9c44;
    border-left: 4px solid #00ff9c;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
}

.verdict-manipulated {
    background: linear-gradient(135deg, #1a0000, #2a0008);
    border: 1px solid #ff4b4b44;
    border-left: 4px solid #ff4b4b;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
}

.verdict-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 6px;
    margin-bottom: 0.5rem;
}

.verdict-authentic .verdict-label { color: #00ff9c; }
.verdict-manipulated .verdict-label { color: #ff4b4b; }

.verdict-confidence {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 2px;
}

/* Metric boxes */
.metric-box {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    color: #ffffff;
    font-weight: 600;
}

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Image preview */
.stImage > img {
    border-radius: 6px;
    border: 1px solid #1e1e2e;
}

/* Progress bar override */
.stProgress > div > div {
    background-color: #4a4aff;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 2px; }

/* Model stats bar */
.stats-bar {
    display: flex;
    gap: 1rem;
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #555;
    letter-spacing: 1px;
}

.stat-item { display: flex; gap: 0.5rem; align-items: center; }
.stat-val { color: #4a4aff; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    classifier = joblib.load("classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("config.json") as f:
        config = json.load(f)
    return clip_model, preprocess, classifier, scaler, config, device

with st.spinner("Initializing terminal..."):
    clip_model, preprocess, classifier, scaler, config, device = load_models()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="terminal-header">// MULTIMODAL ANALYSIS SYSTEM v1.0</p>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">VERITAS</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Visual Misinformation Detection Terminal</p>', unsafe_allow_html=True)

# Model stats bar
st.markdown(f"""
<div class="stats-bar">
    <span class="stat-item">MODEL <span class="stat-val">CLIP ViT-B/32</span></span>
    <span>·</span>
    <span class="stat-item">ACCURACY <span class="stat-val">{config['test_accuracy']*100:.1f}%</span></span>
    <span>·</span>
    <span class="stat-item">F1 <span class="stat-val">{config['test_f1']:.4f}</span></span>
    <span>·</span>
    <span class="stat-item">ROC-AUC <span class="stat-val">{config['test_roc_auc']:.4f}</span></span>
    <span>·</span>
    <span class="stat-item">DATASET <span class="stat-val">FAKEDDIT 140K+</span></span>
    <span>·</span>
    <span class="stat-item">STATUS <span class="stat-val" style="color:#00ff9c">● ONLINE</span></span>
</div>
""", unsafe_allow_html=True)

# ── Input section ──────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<p class="panel-label">// INPUT PARAMETERS</p>', unsafe_allow_html=True)

    headline = st.text_area(
        "NEWS HEADLINE",
        placeholder="Enter the news headline or caption...",
        height=80,
        label_visibility="visible"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="panel-label">// IMAGE SOURCE</p>', unsafe_allow_html=True)

    tab_url, tab_upload = st.tabs(["[ URL ]", "[ UPLOAD ]"])

    image = None

    with tab_url:
        url_input = st.text_input(
            "IMAGE URL",
            placeholder="https://...",
            label_visibility="collapsed"
        )
        if url_input:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
                }
                response = requests.get(url_input, timeout=10, headers=headers)
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    st.error(f"URL did not return an image. Content-Type: {content_type}")
                else:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image: {e}")

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("⟶  RUN ANALYSIS", use_container_width=True)

# ── Analysis & Results ─────────────────────────────────────────────────────────
with col_result:
    st.markdown('<p class="panel-label">// ANALYSIS OUTPUT</p>', unsafe_allow_html=True)

    if analyze_btn:
        if not headline:
            st.warning("Please enter a headline.")
        elif image is None:
            st.warning("Please provide an image via URL or upload.")
        else:
            with st.spinner("Analyzing..."):
                # Extract CLIP embeddings
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                text_tokens = clip.tokenize([headline], truncate=True).to(device)

                with torch.no_grad():
                    img_feat = clip_model.encode_image(image_tensor)
                    txt_feat = clip_model.encode_text(text_tokens)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                img_emb = img_feat.cpu().numpy()
                txt_emb = txt_feat.cpu().numpy()
                cosine_sim = float((img_emb * txt_emb).sum())

                # Build feature vector
                features = np.hstack([img_emb, txt_emb, [[cosine_sim]]])
                features_scaled = scaler.transform(features)

                # Predict
                pred = classifier.predict(features_scaled)[0]
                proba = classifier.predict_proba(features_scaled)[0]
                confidence = float(proba[pred]) * 100
                fake_prob = float(proba[1]) * 100
                real_prob = float(proba[0]) * 100

            # Verdict
            if pred == 0:
                st.markdown(f"""
                <div class="verdict-authentic">
                    <div class="verdict-label">✓ AUTHENTIC</div>
                    <div class="verdict-confidence">CONFIDENCE: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-manipulated">
                    <div class="verdict-label">⚠ MANIPULATED</div>
                    <div class="verdict-confidence">CONFIDENCE: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{real_prob:.1f}%</div>
                    <div class="metric-label">Real Score</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{fake_prob:.1f}%</div>
                    <div class="metric-label">Fake Score</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{cosine_sim:.3f}</div>
                    <div class="metric-label">Image-Text Alignment</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Alignment bar
            st.markdown('<p class="panel-label">// IMAGE-TEXT ALIGNMENT SCORE</p>', unsafe_allow_html=True)
            st.progress(min(max(cosine_sim, 0.0), 1.0))

            st.markdown('<p class="panel-label" style="margin-top:1rem">// PROBABILITY DISTRIBUTION</p>', unsafe_allow_html=True)
            st.progress(real_prob / 100)
            st.markdown(f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;color:#555;letter-spacing:1px">REAL {real_prob:.1f}% ←——→ FAKE {fake_prob:.1f}%</p>', unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            height: 400px;
            border: 1px dashed #1e1e2e;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #2a2a3e;
            font-family: 'Share Tech Mono', monospace;
            font-size: 12px;
            letter-spacing: 2px;
        ">
            <div style="font-size:2rem;margin-bottom:1rem">⟳</div>
            <div>AWAITING INPUT</div>
            <div style="font-size:10px;margin-top:0.5rem;color:#222">ENTER HEADLINE + IMAGE TO BEGIN</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    border-top: 1px solid #1e1e2e;
    padding-top: 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #2a2a3e;
    letter-spacing: 1px;
    display: flex;
    justify-content: space-between;
">
    <span>VERITAS // VISUAL MISINFORMATION DETECTION</span>
    <span>CLIP ViT-B/32 + MLP · FAKEDDIT DATASET</span>
</div>
""", unsafe_allow_html=True)