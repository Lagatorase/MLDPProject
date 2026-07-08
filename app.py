import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

st.set_page_config(
    page_title="East Region HDB Price Estimator",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
    <style>
    .hero {
        background: linear-gradient(135deg, #1a3c5e 0%, #2e8b57 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #e0e8e4;
        font-size: 1.05rem;
        margin-top: 0.3rem;
    }
    .price-box {
        background-color: #f0f7f4;
        border-left: 6px solid #2e8b57;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f7f8fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = joblib.load("hdb_resale_model.pkl")
    columns = joblib.load("model_columns.pkl")
    stats = joblib.load("app_summary_stats.pkl")
    return model, columns, stats

model, model_columns, stats = load_resources()

st.markdown("""
    <div class="hero">
        <p class="hero-title">East Region HDB Resale Price Estimator</p>
        <p class="hero-subtitle">Instant, data-driven price estimates for flats in Bedok, Pasir Ris, and Tampines</p>
    </div>
""", unsafe_allow_html=True)

TOWN_OPTIONS = ["BEDOK", "PASIR RIS", "TAMPINES"]
FLAT_TYPE_OPTIONS = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
FLAT_MODEL_OPTIONS = [
    "2-room", "3Gen", "Adjoined flat", "Apartment", "DBSS", "Improved",
    "Maisonette", "Model A", "Model A-Maisonette", "Multi Generation",
    "New Generation", "Premium Apartment", "Premium Maisonette",
    "Simplified", "Standard"
]

col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.subheader("Flat Details")

    town = st.selectbox("Town", TOWN_OPTIONS)
    flat_type = st.selectbox("Flat Type", FLAT_TYPE_OPTIONS)
    flat_model = st.selectbox("Flat Model", FLAT_MODEL_OPTIONS)
    floor_area_sqm = st.slider("Floor Area (sqm)", min_value=38, max_value=200, value=95)
    storey_mid = st.slider("Storey Level", min_value=1, max_value=50, value=8)
    lease_commence_year = st.number_input(
        "Lease Commencement Year", min_value=1972, max_value=2022, value=1995, step=1
    )

    predict_clicked = st.button("Get Price Estimate", type="primary", use_container_width=True)

    with st.expander("How this estimate works"):
        st.write(
            "This tool uses a Random Forest model trained on real HDB resale "
            "transactions from 2017 onwards in Bedok, Pasir Ris, and Tampines. "
            "The estimate is a typical price for flats with similar characteristics. "
            "The range shown reflects how much the model's underlying trees "
            "disagree with each other, wider ranges mean less certainty, usually "
            "for less common flat configurations."
        )

with col_right:
    st.subheader("Estimated Resale Price")

    if predict_clicked:
        today = datetime.now()
        txn_year = today.year
        txn_month = today.month

        if lease_commence_year > txn_year:
            st.error("Lease commencement year cannot be in the future. Please check your input.")
        else:
            try:
                remaining_lease_years = 99 - (txn_year - lease_commence_year)

                input_row = pd.DataFrame(0, index=[0], columns=model_columns)
                input_row["floor_area_sqm"] = floor_area_sqm
                input_row["remaining_lease_years"] = remaining_lease_years
                input_row["storey_mid"] = storey_mid
                input_row["txn_year"] = txn_year
                input_row["txn_month"] = txn_month

                for col, val in [("town", town), ("flat_type", flat_type), ("flat_model", flat_model)]:
                    dummy_col = f"{col}_{val}"
                    if dummy_col in input_row.columns:
                        input_row[dummy_col] = 1

                prediction = model.predict(input_row)[0]

                # Price range from spread across individual trees in the forest
                tree_preds = [tree.predict(input_row)[0] for tree in model.estimators_]
                lower = np.percentile(tree_preds, 10)
                upper = np.percentile(tree_preds, 90)

                price_per_sqm = prediction / floor_area_sqm
                town_avg = stats['town_avg_price'].get(town, prediction)
                diff_pct = ((prediction - town_avg) / town_avg) * 100

                st.session_state["prediction"] = prediction
                st.session_state["lower"] = lower
                st.session_state["upper"] = upper
                st.session_state["remaining_lease"] = remaining_lease_years
                st.session_state["price_per_sqm"] = price_per_sqm
                st.session_state["town"] = town
                st.session_state["flat_type"] = flat_type
                st.session_state["diff_pct"] = diff_pct

            except Exception as e:
                st.error(f"Something went wrong generating the estimate: {e}")

    if "prediction" in st.session_state:
        st.markdown(f"""
            <div class="price-box">
                <p style="margin:0; color:#5a6b7d;">Estimated Resale Price</p>
                <p style="margin:0; font-size:2.2rem; font-weight:700; color:#1a3c5e;">
                    ${st.session_state['prediction']:,.0f}
                </p>
                <p style="margin:0.3rem 0 0 0; color:#5a6b7d; font-size:0.9rem;">
                    Typical range: ${st.session_state['lower']:,.0f} to ${st.session_state['upper']:,.0f}
                </p>
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Price per sqm", f"${st.session_state['price_per_sqm']:,.0f}")
        with c2:
            st.metric("Remaining Lease", f"{st.session_state['remaining_lease']:.0f} yrs")
        with c3:
            st.metric(f"vs {st.session_state['town']} Average",
                      f"{st.session_state['diff_pct']:+.1f}%")

        st.divider()
        st.caption(f"Average Prices Across East Region ({st.session_state['flat_type']})")

        flat_type_sel = st.session_state['flat_type']
        town_sel = st.session_state['town']
        chart_data = {
            t: stats['town_flattype_avg_price'].get(t, {}).get(flat_type_sel, 0)
            for t in TOWN_OPTIONS
        }
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#2e8b57' if t == town_sel else '#c8d6cf' for t in chart_data.keys()]
        ax.bar(chart_data.keys(), chart_data.values(), color=colors)
        ax.set_ylabel("Average Resale Price (SGD)")
        st.pyplot(fig)

        st.caption(f"Price Trend Over Time in {town_sel}")
        year_data = stats['town_year_avg_price'].get(town_sel, {})
        years = sorted(year_data.keys())
        prices = [year_data[y] for y in years]
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        ax2.plot(years, prices, marker='o', color='#1a3c5e')
        ax2.set_ylabel("Avg Resale Price (SGD)")
        st.pyplot(fig2)

    else:
        st.info("Fill in the flat details and click **Get Price Estimate** to see a prediction.")

st.divider()
st.caption("Estimates are generated by a Random Forest model trained on HDB resale transactions from 2017 onwards, restricted to Bedok, Pasir Ris, and Tampines.")