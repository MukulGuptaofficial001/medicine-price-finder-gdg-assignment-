import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Medicine Price Finder",
    page_icon="💊",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .price-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(100, 180, 255, 0.2);
        border-radius: 16px;
        padding: 24px 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .price-label {
        font-size: 13px;
        color: #a0aec0;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .price-value {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .medicine-title {
        font-size: 20px;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 8px;
    }

    .comp-badge {
        display: inline-block;
        background: rgba(79, 172, 254, 0.12);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        color: #4facfe;
        margin-top: 10px;
    }

    .alt-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: #0f0c29;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-size: 15px;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(100,180,255,0.25);
        border-radius: 10px;
        color: white;
        font-size: 15px;
        padding: 12px 16px;
    }

    .section-header {
        font-size: 17px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .savings-pill {
        background: rgba(72, 199, 142, 0.15);
        border: 1px solid rgba(72, 199, 142, 0.3);
        color: #48c78e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 💊 Medicine Price & Alternative Finder")
st.markdown("Find the fair price of any medicine and discover cheaper generics with the same composition.")

search_col, btn_col = st.columns([4, 1])
with search_col:
    m_name = st.text_input("", placeholder="e.g. Paracetamol, Dolo, Metformin...", label_visibility="collapsed")
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("Search")

if search_btn and m_name.strip():
    with st.spinner("Fetching data..."):
        pred_res = None
        alt_res = None
        error_msg = None

        try:
            r1 = requests.get(f"{API_BASE}/predict-price", params={"medicine_name": m_name}, timeout=10)
            if r1.status_code == 200:
                pred_res = r1.json()
            else:
                error_msg = r1.json().get("detail", "Medicine not found")
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to the API. Make sure the FastAPI server is running on port 8000."

        if pred_res:
            try:
                r2 = requests.get(f"{API_BASE}/alternatives", params={"medicine_name": m_name}, timeout=10)
                if r2.status_code == 200:
                    alt_res = r2.json()
            except Exception:
                pass

        if error_msg:
            st.error(f"❌ {error_msg}")
        elif pred_res:
            st.markdown(f"""
                <div class="price-card">
                    <div class="price-label">Predicted Fair Price</div>
                    <div class="price-value">₹{pred_res['predicted_price']:.2f}</div>
                    <div class="medicine-title">{pred_res['medicine_name']}</div>
                    <div style="color:#a0aec0; font-size:13px; margin-top:4px;">{pred_res['manufacturer']}</div>
                    <div class="comp-badge">{pred_res['composition'][:60]}{'...' if len(pred_res['composition']) > 60 else ''}</div>
                </div>
            """, unsafe_allow_html=True)

            if alt_res and alt_res.get("alternatives"):
                alts = alt_res["alternatives"]

                st.markdown('<div class="section-header">📊 Price Comparison by Manufacturer</div>', unsafe_allow_html=True)

                chart_data = [{"Medicine": pred_res["medicine_name"], "Manufacturer": pred_res["manufacturer"], "Price": pred_res["predicted_price"]}]
                for a in alts:
                    chart_data.append({"Medicine": a["medicine_name"], "Manufacturer": a["manufacturer"], "Price": a["price"]})
                chart_df = pd.DataFrame(chart_data)

                fig = px.bar(
                    chart_df,
                    x="Manufacturer",
                    y="Price",
                    color="Price",
                    text="Price",
                    color_continuous_scale=["#48c78e", "#4facfe", "#f0a500", "#f25f5c"],
                    template="plotly_dark"
                )
                fig.update_traces(texttemplate="₹%{text:.0f}", textposition="outside")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    coloraxis_showscale=False,
                    yaxis_title="Price (₹)",
                    xaxis_title="",
                    margin=dict(t=20, b=20),
                    height=320
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-header">🏷️ Cheaper Generic Alternatives</div>', unsafe_allow_html=True)

                for i, alt in enumerate(alts[:5]):
                    savings = pred_res["predicted_price"] - alt["price"]
                    savings_str = f"Save ₹{savings:.0f}" if savings > 0 else "Similar price"
                    pill_color = "savings-pill" if savings > 0 else ""
                    st.markdown(f"""
                        <div class="alt-card">
                            <div>
                                <div style="font-weight:600; color:#e2e8f0; font-size:15px;">#{i+1} {alt['medicine_name']}</div>
                                <div style="color:#a0aec0; font-size:12px; margin-top:2px;">{alt['manufacturer']}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:20px; font-weight:700; color:#4facfe;">₹{alt['price']:.2f}</div>
                                <div class="{pill_color}" style="font-size:11px;">{savings_str}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No cheaper alternatives found with the same composition in the dataset.")

elif search_btn and not m_name.strip():
    st.warning("Please enter a medicine name to search.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#4a5568; font-size:12px;'>Built for GDG ML Assignment · Data: Indian Pharmaceutical Products (Kaggle)</div>", unsafe_allow_html=True)
