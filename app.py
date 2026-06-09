import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
    }
    .prediction-label {
        font-size: 1rem;
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f2937;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e5e7eb;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train Model (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("HousePricePrediction.csv")

    # Preprocessing
    df["SalePrice"] = df["SalePrice"].fillna(df["SalePrice"].mean())
    df = df.dropna()

    cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
    df_encoded = pd.get_dummies(df, columns=cols, drop_first=True)

    X = df_encoded.drop(["SalePrice", "Id"], axis=1)
    y = df_encoded["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "R² Score":  round(r2_score(y_test, y_pred), 4),
        "MAE":       round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE":      round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }

    return model, scaler, X.columns.tolist(), metrics, df


model, scaler, feature_cols, metrics, raw_df = load_and_train()


# ─── Prediction Helper ────────────────────────────────────────────────────────
def build_input_row(user_inputs: dict) -> pd.DataFrame:
    """Build a one-row DataFrame that matches the training feature columns."""
    row = {col: 0 for col in feature_cols}

    # Numeric fields
    numeric_fields = [
        "MSSubClass", "LotArea", "OverallCond",
        "YearBuilt", "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"
    ]
    for f in numeric_fields:
        if f in row:
            row[f] = user_inputs[f]

    # Encoded categorical fields
    cat_map = {
        "MSZoning":   "MSZoning",
        "LotConfig":  "LotConfig",
        "BldgType":   "BldgType",
        "Exterior1st":"Exterior1st",
    }
    for field, prefix in cat_map.items():
        val = user_inputs[field]
        encoded_col = f"{prefix}_{val}"
        if encoded_col in row:
            row[encoded_col] = 1

    return pd.DataFrame([row])


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Linear Regression model trained on Ames Housing Dataset · '
    'Enter property details to get an instant price estimate</div>',
    unsafe_allow_html=True
)

# ─── Sidebar — Model Info ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Performance")
    for k, v in metrics.items():
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.8rem">
            <div class="metric-title">{k}</div>
            <div class="metric-value">{v:,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Dataset")
    st.info(f"**{len(raw_df):,}** records · **{raw_df.shape[1]}** features")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Built with **Scikit-learn** + **Streamlit**  \n"
        "Model: Linear Regression w/ StandardScaler  \n"
        "By **Gyanendra Chauhan**"
    )

# ─── Input Form ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="section-header">🏗️ Building Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        ms_subclass = st.selectbox(
            "Building Class (MSSubClass)",
            [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
            index=0,
            help="Type of dwelling involved in the sale"
        )
        bldg_type = st.selectbox(
            "Building Type",
            ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
            help="Type of dwelling"
        )
        year_built = st.slider("Year Built", 1872, 2010, 1990)

    with c2:
        ms_zoning = st.selectbox(
            "Zoning Classification",
            ["RL", "RM", "C (all)", "FV", "RH"],
            help="General zoning classification"
        )
        exterior = st.selectbox(
            "Exterior Material",
            ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "BrkFace",
             "WdShing", "CemntBd", "Plywood", "AsbShng", "Stucco",
             "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"],
        )
        year_remod = st.slider("Year Remodeled", 1950, 2010, 2000)

    st.markdown('<div class="section-header">📐 Area & Lot Details</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        lot_area = st.number_input(
            "Lot Area (sq ft)", min_value=1300, max_value=215245, value=8000, step=100
        )
        lot_config = st.selectbox(
            "Lot Configuration",
            ["Inside", "FR2", "Corner", "CulDSac", "FR3"]
        )
    with c4:
        total_bsmt = st.number_input(
            "Total Basement SF", min_value=0, max_value=6110, value=800, step=50
        )
        bsmt_fin_sf2 = st.number_input(
            "Basement Finished SF (Type 2)", min_value=0, max_value=1526, value=0, step=10
        )

    st.markdown('<div class="section-header">⭐ Condition</div>', unsafe_allow_html=True)
    overall_cond = st.slider(
        "Overall Condition (1 = Very Poor → 9 = Excellent)", 1, 9, 5
    )

with col_right:
    # Predict on button click
    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

    if predict_btn:
        user_inputs = {
            "MSSubClass": ms_subclass,
            "MSZoning": ms_zoning,
            "LotArea": lot_area,
            "LotConfig": lot_config,
            "BldgType": bldg_type,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "Exterior1st": exterior,
            "BsmtFinSF2": bsmt_fin_sf2,
            "TotalBsmtSF": total_bsmt,
        }

        input_df = build_input_row(user_inputs)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Sale Price</div>
            <div class="prediction-value">${prediction:,.0f}</div>
            <div style="font-size:0.85rem;opacity:0.75;margin-top:0.5rem">
                ± based on model MAE of ${metrics['MAE']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Price range
        low = max(0, prediction - metrics["MAE"])
        high = prediction + metrics["MAE"]
        st.markdown(f"**Likely Range:** `${low:,.0f}` — `${high:,.0f}`")

        st.markdown('<div class="section-header">📋 Your Input Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Feature": list(user_inputs.keys()),
            "Value": list(user_inputs.values())
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Fill in the property details on the left, then click **Predict Price**.")

        st.markdown('<div class="section-header">📈 Price Distribution (Sample)</div>', unsafe_allow_html=True)
        sample = raw_df["SalePrice"].dropna().sample(min(500, len(raw_df)), random_state=42)
        st.bar_chart(sample.value_counts(bins=20).sort_index())

# ─── Dataset Explorer (Expander) ─────────────────────────────────────────────
with st.expander("🔍 Explore Raw Dataset"):
    n = st.slider("Rows to show", 5, 50, 10)
    st.dataframe(raw_df.head(n), use_container_width=True)
