import streamlit as st
import time
from utils import predict_url, load_model

# Page Configuration
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Techy" Dark Blue Theme
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
    .logo-container { display: flex; align-items: center; gap: 15px; margin-bottom: 30px; }
    .neon-shield { 
        width: 60px; height: 60px; background-color: #020617; border: 2px solid #06b6d4; 
        border-radius: 15px; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
    }
    .stTextInput input { background-color: #020617 !important; color: #f8fafc !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
    .stButton button { background-color: #06b6d4 !important; color: white !important; border-radius: 12px !important; font-weight: bold !important; border: none !important; padding: 10px 20px !important; }
    .result-card { padding: 30px; border-radius: 24px; border: 2px solid rgba(255, 255, 255, 0.1); margin-top: 20px; }
    .phishing-detected { background-color: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3); }
    .safe-url { background-color: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="neon-shield">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m12 8-4 4 4 4 4-4-4-4Z"/></svg>
        </div>
        <h1 style="margin:0; font-size: 24px; font-weight: 900; color: white;">PhishGuard AI</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Project Status")
    model, scaler, demo_mode = load_model()
    if demo_mode:
        st.warning("⚠️ Running in Demo Mode. Please place 'model.pkl' and 'scaler.pkl' in the /models folder for live AI prediction.")
    else:
        st.success("✅ AI Model Loaded Successfully")
    
    st.markdown("---")
    st.subheader("Example URLs")
    examples = [
        "https://secure-login.paypal-verify.com",
        "https://www.google.com",
        "http://192.168.1.1/login.html",
        "https://github.com/trending"
    ]
    
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.url_input = ex
            st.rerun()

    st.markdown("---")
    st.caption("AI Mini-Project 2026 • Internal Use Only")

# Main Content
st.title("Phishing Detection Dashboard")
st.write("Enter a URL below to perform a real-time security analysis using our XGBoost classifier.")

if 'url_input' not in st.session_state:
    st.session_state.url_input = ""

url = st.text_input("URL Input", value=st.session_state.url_input, placeholder="https://example.com", label_visibility="collapsed")

if st.button("Analyze URL") and url:
    with st.status("Analyzing URL...", expanded=True) as status:
        st.write("Extracting 11 heuristic features...")
        time.sleep(0.3)
        is_phishing, confidence, features = predict_url(url, model, scaler, demo_mode)
        st.write("Generating security report...")
        time.sleep(0.2)
        status.update(label="Analysis Complete", state="complete", expanded=False)
    
    if is_phishing:
        st.markdown(f"""
        <div class="result-card phishing-detected">
            <h2 style="color: #ef4444; margin-top: 0;">🚨 Phishing Detected</h2>
            <p style="font-size: 1.2rem;">Risk Level: <b>HIGH</b></p>
            <p>AI Confidence: <b>{confidence:.2f}%</b></p>
            <hr style="opacity: 0.1; margin: 20px 0;">
            <p style="font-size: 0.8rem; opacity: 0.7; font-family: monospace;">TARGET: {url}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card safe-url">
            <h2 style="color: #10b981; margin-top: 0;">✅ URL Appears Safe</h2>
            <p style="font-size: 1.2rem;">Risk Level: <b>LOW</b></p>
            <p>AI Confidence: <b>{confidence:.2f}%</b></p>
            <hr style="opacity: 0.1; margin: 20px 0;">
            <p style="font-size: 0.8rem; opacity: 0.7; font-family: monospace;">TARGET: {url}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Detailed Feature Breakdown"):
        st.write("The following 11 features were extracted and passed to the model:")
        cols = st.columns(3)
        feature_list = list(features.items())
        for i, (name, val) in enumerate(feature_list):
            with cols[i % 3]:
                st.metric(name, f"{val:.4f}" if isinstance(val, float) else val)

elif not url:
    st.markdown("""
    <div style="text-align: center; padding: 100px 0; opacity: 0.3;">
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m12 8-4 4 4 4 4-4-4-4Z"/></svg>
        <h3>System Ready</h3>
        <p>Paste a URL above to initiate the security scan.</p>
    </div>
    """, unsafe_allow_html=True)
