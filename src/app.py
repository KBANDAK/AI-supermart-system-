import streamlit as st
import requests
from PIL import Image

# 1. PAGE SETUP
st.set_page_config(page_title="FreshMarket Scanner", page_icon="🛒", layout="centered")

# 2. DESIGN: Force High-Contrast Black Text
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .stButton>button { 
        background-color: #28a745; color: white; height: 3em; 
        font-size: 20px; font-weight: bold; border-radius: 10px;
    }
    .receipt-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-top: 8px solid #dc3545;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-family: 'Courier New', monospace;
        color: #000000 !important;
        margin-top: 20px;
    }
    /* Ensure all text inside receipt is strictly BLACK */
    .receipt-box h3, .receipt-box p, .receipt-box div, .receipt-box strong {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 FreshMarket")
st.markdown("<h4 style='text-align: center;'>Instant AI Price Checker</h4>", unsafe_allow_html=True)

# 3. SCANNING INTERFACE
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True)
    
    if st.button("🔍 CHECK PRICE NOW"):
        with st.spinner("Accessing Store Database..."):
            try:
                # Production URL (NO '-test')
                N8N_URL = "http://localhost:5678/webhook-test/scan"
                
                files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                response = requests.post(N8N_URL, files=files, data={"text": "scan"})
                
                if response.status_code == 200:
                    res = response.json()
                    st.balloons()
                    
                    # THE RECEIPT DISPLAY
                    st.markdown(f"""
                    <div class="receipt-box">
                        <h3 style="text-align: center;">🧾 OFFICIAL RECEIPT</h3>
                        <hr style="border-top: 2px dashed #000;">
                        <p><strong>SCAN RESULT:</strong> {res.get('seen', 'Unknown')}</p>
                        <p><strong>AVAILABILITY:</strong> In Stock</p>
                        <br>
                        <div style="background-color: #f8f9fa; padding: 15px; border: 1px solid #000;">
                            <strong>STORE ASSISTANT:</strong><br>
                            {res.get('reply', 'No data found.')}
                        </div>
                        <hr style="border-top: 2px dashed #000;">
                        <p style="text-align: center; font-size: 14px;">THANK YOU FOR SHOPPING!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Server Busy (Error {response.status_code}). Ensure n8n is 'PUBLISHED'.")
            except Exception as e:
                st.error("⚠️ Connection Lost. Check if Python and n8n are running.")