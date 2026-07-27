import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import base64
from datetime import datetime

st.set_page_config(
    page_title="East Region HDB Price Estimator",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("hdb_background.jpg")

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(10, 15, 20, 0.88), rgba(10, 20, 18, 0.90)),
                    url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .main .block-container {{
        background-color: rgba(18, 24, 30, 0.75);
        border-radius: 16px;
        padding: 2rem 3rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0;
    }}
    .hero-subtitle {{
        color: #a8b8c8;
        font-size: 1.1rem;
        margin-top: 0.4rem;
    }}
    .price-box {{
        background-color: rgba(46, 139, 87, 0.5);
        border-left: 6px solid #3fd67a;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }}
    .price-box p {{
        color: #e8f0ec !important;
    }}
    h1, h2, h3, p, label, .stMarkdown, .stCaption {{
        color: #e8edf2 !important;
    }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = joblib.load("hdb_resale_model.pkl")
    columns = joblib.load("model_columns.pkl")
    stats = joblib.load("app_summary_stats.pkl")
    return model, columns, stats

model, model_columns, stats = load_resources()

st.markdown('<p class="hero-title">East Region HDB Resale Price Estimator</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Instant, data-driven price estimates for flats in Bedok, Pasir Ris, and Tampines</p>', unsafe_allow_html=True)
st.write("")

with st.expander("Learn more about the data and model behind this estimate"):
    st.markdown("""
    **Data source:** Official HDB resale flat transaction records from data.gov.sg,
    covering transactions registered from January 2017 onwards, restricted to the
    East Region towns of Bedok, Pasir Ris, and Tampines.

    **What the model learned from:** Each flat's town, flat type, flat model, floor
    area, storey level, remaining lease, and transaction timing, matched against its
    actual resale price.

    **How it works:** A Random Forest model, an ensemble of many decision trees that
    each learn different patterns in the data and vote together on a final estimate.
    This approach handles the real-world irregularities in resale pricing (different
    flat models, uneven transaction volumes across towns) better than a single
    straight-line model would.

    **Model performance:** On transactions the model had never seen before, estimates
    were typically within about $25,200 of the actual sale price, explaining roughly
    96% of the price variation across all three towns.

    **Limitations:** Estimates are based on historical patterns and are meant as a
    starting reference point, not a professional valuation. Rare flat models and
    less common towns (fewer historical transactions) may have wider uncertainty.
    """)

st.divider()

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

                tree_preds = [tree.predict(input_row.values)[0] for tree in model.estimators_]
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
                <p style="margin:0; color:#a8b8c8;">Estimated Resale Price</p>
                <p style="margin:0; font-size:2.2rem; font-weight:700; color:#ffffff;">
                    ${st.session_state['prediction']:,.0f}
                </p>
                <p style="margin:0.3rem 0 0 0; color:#a8b8c8; font-size:0.9rem;">
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

        plt.style.use('dark_background')

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        colors = ['#3fd67a' if t == town_sel else '#4a5568' for t in chart_data.keys()]
        ax.bar(chart_data.keys(), chart_data.values(), color=colors)
        ax.set_ylabel("Average Resale Price (SGD)")
        st.pyplot(fig)

        st.caption(f"Price Trend Over Time in {town_sel}")
        year_data = stats['town_year_avg_price'].get(town_sel, {})
        years = sorted(year_data.keys())
        prices = [year_data[y] for y in years]

        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor('none')
        ax2.plot(years, prices, marker='o', color='#3fd67a')
        ax2.set_ylabel("Avg Resale Price (SGD)")
        st.pyplot(fig2)

    else:
        st.info("Fill in the flat details and click **Get Price Estimate** to see a prediction.")

st.divider()
st.caption("Estimates are generated by a Random Forest model trained on HDB resale transactions from 2017 onwards, restricted to Bedok, Pasir Ris, and Tampines. For guidance only, not a substitute for professional valuation.")