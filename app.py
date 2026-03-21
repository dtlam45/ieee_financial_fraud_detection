"""
IEEE Fraud Detection — Streamlit Demo
  - Load XGBoost booster (.ubj) + uid_agg + freq_maps
  - Feature engineering: UID, frequency encoding, extra features
  - Predict single transaction hoặc upload CSV batch
  - EDA charts: label distribution, TransactionAmt, ProductCD fraud rate, correlation heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os


st.set_page_config(
    page_title="IEEE Financial Fraud Detection - Team ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  h1, h2, h3 { font-family: 'Courier New', monospace; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 16px 20px; margin-bottom: 8px;
  }
  .fraud-badge {
    background: rgba(248,81,73,0.15); border: 1px solid #f85149;
    color: #f85149; padding: 4px 12px; border-radius: 4px;
    font-family: monospace; font-weight: bold; font-size: 18px;
  }
  .legit-badge {
    background: rgba(63,185,80,0.15); border: 1px solid #3fb950;
    color: #3fb950; padding: 4px 12px; border-radius: 4px;
    font-family: monospace; font-weight: bold; font-size: 18px;
  }
  .warn-badge {
    background: rgba(210,153,34,0.15); border: 1px solid #d2993a;
    color: #d2993a; padding: 4px 12px; border-radius: 4px;
    font-family: monospace; font-weight: bold; font-size: 18px;
  }
  .stProgress > div > div { background: #3fb950; }
</style>
""", unsafe_allow_html=True)

CAT_COLS = [
    'ProductCD', 'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'M4', 'M6', 'id_30', 'id_31', 'DeviceType',
]
_EXCLUDE_COLS = {
    "TransactionID", "isFraud", "uid", "D1_norm",
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M4", "M6", "DeviceType", "id_30", "id_31", "DeviceInfo", "id_33",
}

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


@st.cache_data
def load_uid_agg(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_pickle(path)


@st.cache_data
def load_freq_maps(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── FEATURE ENGINEERING (mirror chính xác notebook) ────────────────────────
def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUG FIX: notebook dùng F.round(..., scale=0) tức là round to nearest integer
    pandas .round(0) trả về float, cần .astype(int) để concat đúng như Spark
    """
    df = df.copy()
    # Phải cast sang int sau khi round để tạo uid string giống hệt Spark
    df['D1_norm'] = (df['TransactionDT'] / 86400 - df['D1']).round(0).astype('Int64')
    df['uid'] = (
        df['card1'].astype(str) + '_' +
        df['addr1'].astype(str) + '_' +
        df['D1_norm'].astype(str)
    )
    return df


def apply_uid_agg(df: pd.DataFrame, uid_agg: pd.DataFrame) -> pd.DataFrame:
    # BUG FIX: drop cột trùng trước khi merge để tránh suffix _x _y
    uid_cols = [c for c in uid_agg.columns if c != 'uid']
    df = df.drop(columns=[c for c in uid_cols if c in df.columns], errors='ignore')
    df = df.merge(uid_agg, on='uid', how='left')
    df['uid_amt_std'] = df['uid_amt_std'].fillna(0.0)
    return df


def apply_freq_encoding(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    df = df.copy()
    for col_name, freq_df in freq_maps.items():
        if col_name not in df.columns:
            continue

        freq_col = f'{col_name}_freq'

        # BUG FIX: drop cột _freq cũ nếu đã tồn tại (tránh _freq_x, _freq_y)
        if freq_col in df.columns:
            df = df.drop(columns=[freq_col])

        freq_df = freq_df.copy()
        df[col_name]      = df[col_name].astype(str)
        freq_df[col_name] = freq_df[col_name].astype(str)

        df = df.merge(
            freq_df[[col_name, freq_col]],
            on=col_name, how='left'
        )
        df[freq_col] = df[freq_col].fillna(0)

        # Khôi phục NaN cho categorical gốc
        df[col_name] = df[col_name].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['tx_hour']           = ((df['TransactionDT'] / 3600) % 24).astype(int)
    df['tx_day_of_week']    = ((df['TransactionDT'] / 86400) % 7).astype(int)
    df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
    df['amt_to_uid_mean_ratio'] = np.where(
        df['uid_amt_mean'].notna() & (df['uid_amt_mean'] > 0),
        df['TransactionAmt'] / df['uid_amt_mean'],
        1.0
    )
    return df


def run_full_pipeline(
    df: pd.DataFrame,
    uid_agg: pd.DataFrame,
    freq_maps: dict,
    booster: xgb.Booster,
) -> pd.DataFrame:
    """
    BUG FIX CHÍNH: Không dùng FEATURE_COLS_ORDER hardcode.
    Lấy feature_names trực tiếp từ booster (đúng thứ tự lúc train),
    rồi select + reindex để align chính xác.
    """
    df = add_uid(df)
    df = apply_uid_agg(df, uid_agg)
    df = apply_freq_encoding(df, freq_maps)
    df = add_extra_features(df)

    # Lấy feature names từ booster — source of truth duy nhất
    feature_names = booster.feature_names  # có thể là None nếu model train không set names

    if feature_names is not None:
        # Happy path: booster biết tên features → align chính xác
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan
        X = df[feature_names].astype(np.float32)
    else:
        # Fallback: booster không có feature names (train bằng numpy array thuần)
        # → loại các cột non-numeric và exclude cols, giữ nguyên thứ tự như notebook
        numeric_cols = [
            c for c in df.columns
            if c not in _EXCLUDE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[numeric_cols].astype(np.float32)

    return X


def predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    if booster.feature_names is not None:
        # Model có feature names → truyền vào để XGBoost validate thứ tự
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    else:
        # Model train bằng numpy thuần, không có feature names → truyền array
        dmat = xgb.DMatrix(X.values)
    return booster.predict(dmat)


_EMAIL_RISK  = {'anonymous.com': 0.82, 'unknown.xyz': 0.76, 'protonmail.com': 0.32,
                'yahoo.com': 0.15, 'gmail.com': 0.07, 'hotmail.com': 0.10, 'other': 0.18}
_PRODUCT_RISK = {'C': 0.68, 'H': 0.42, 'R': 0.28, 'S': 0.20, 'W': 0.14}
_CARD_RISK    = {'discover': 0.20, 'american express': 0.14, 'mastercard': 0.10, 'visa': 0.08}

def demo_predict(row: dict) -> float:
    score = 0.05
    score += _EMAIL_RISK.get(row.get('P_emaildomain', 'other'), 0.18) * 0.30
    score += _PRODUCT_RISK.get(row.get('ProductCD', 'W'), 0.14) * 0.25
    score += _CARD_RISK.get(row.get('card4', 'visa'), 0.08) * 0.15
    amt = float(row.get('TransactionAmt', 100))
    log_amt = np.log1p(amt)
    if abs(log_amt - 4.0) < 0.8 or abs(log_amt - 5.3) < 0.6:
        score += 0.10
    if amt < 5:
        score += 0.14
    if amt > 2000:
        score += 0.08
    hour = int(row.get('tx_hour', 14))
    if hour <= 5:
        score += 0.12
    elif hour >= 22:
        score += 0.06
    c1 = float(row.get('C1', 1))
    if c1 > 10:
        score += 0.08
    score += (np.random.rand() - 0.5) * 0.03
    return float(np.clip(score, 0.01, 0.97))


with st.sidebar:
    st.markdown("## Team LHAL")
    st.markdown("**IEEE-CIS Financial Fraud Detection**")
    st.markdown("---")

    st.markdown("### 📂 Load Artifacts")

    model_path = st.text_input(
        "XGBoost model path",
        value="./best_xgb_model/best_booster.ubj",
    )
    uid_agg_path = st.text_input(
        "uid_agg path",
        value="./artifacts/uid_agg.parquet",
    )
    freq_maps_path = st.text_input(
        "freq_maps path",
        value="./artifacts/freq_maps.pkl",
    )
    threshold = st.slider(
        "Fraud threshold", min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
    )

    if st.button("🔄 Load Model", use_container_width=True):
        if os.path.exists(model_path):
            try:
                st.session_state['booster'] = load_model(model_path)
                fn = st.session_state['booster'].feature_names or []
                n_feat = len(fn) if fn else st.session_state['booster'].num_features()
                st.success(f"Model loaded! ({n_feat} features)")
            except Exception as e:
                st.error(f"{e}")
                st.session_state['booster'] = None
        else:
            st.warning("⚠ Model file not found. Using demo mode.")
            st.session_state['booster'] = None

        if os.path.exists(uid_agg_path):
            st.session_state['uid_agg'] = load_uid_agg(uid_agg_path)
            st.success(f"uid_agg loaded! ({len(st.session_state['uid_agg']):,} UIDs)")
        else:
            st.session_state['uid_agg'] = None
            st.warning("uid_agg not found")

        if os.path.exists(freq_maps_path):
            st.session_state['freq_maps'] = load_freq_maps(freq_maps_path)
            st.success(f"freq_maps loaded! ({len(st.session_state['freq_maps'])} cols)")
        else:
            st.session_state['freq_maps'] = None
            st.warning("freq_maps not found")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Status")
    booster_ok = st.session_state.get('booster') is not None
    uid_ok     = st.session_state.get('uid_agg') is not None
    freq_ok    = st.session_state.get('freq_maps') is not None

    st.markdown(f"{'🟢' if booster_ok else '🔴'} XGBoost Booster")
    st.markdown(f"{'🟢' if uid_ok else '🟡'} uid_agg (optional)")
    st.markdown(f"{'🟢' if freq_ok else '🟡'} freq_maps (optional)")

    if booster_ok:
        fn = st.session_state['booster'].feature_names or []
        n_feat = len(fn) if fn else st.session_state['booster'].num_features()
        with st.expander(f"📋 Feature names ({n_feat})"):
            if fn:
                st.text('\n'.join(fn))
            else:
                st.info(f"Model có {n_feat} features nhưng không lưu tên (trained without feature names).")

    st.markdown("---")
    st.markdown("### 📖 Model Config")
    st.code("""
n_estimators : 500
max_depth    : 6
learning_rate: 0.05
subsample    : 0.8
colsample    : 0.8
scale_pos_wt : ~6.2
split        : time-based
    """, language="text")


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_eda = st.tabs([
    "🔮 Predict Transaction",
    "📋 Batch Scoring",
    "📊 EDA",
])

with tab_predict:
    st.markdown("## 🔮 Single Transaction Scoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💳 Transaction Info**")
        tx_amt     = st.number_input("TransactionAmt ($)", min_value=0.01, value=125.50, step=0.01)
        product_cd = st.selectbox("ProductCD", ['W', 'H', 'C', 'S', 'R'])
        card4      = st.selectbox("card4 (Network)", ['visa', 'mastercard', 'discover', 'american express'])
        card6      = st.selectbox("card6 (Type)", ['debit', 'credit', 'charge card'])
        p_email    = st.selectbox("P_emaildomain", [
            'gmail.com', 'yahoo.com', 'hotmail.com',
            'anonymous.com', 'protonmail.com', 'unknown.xyz', 'other'
        ])

    with col2:
        st.markdown("**🏦 Card & Address**")
        card1 = st.number_input("card1 (Bank ID)", min_value=0, max_value=20000, value=9500)
        card2 = st.number_input("card2", min_value=0, max_value=1000, value=321)
        card3 = st.number_input("card3", min_value=0, max_value=200, value=150)
        card5 = st.number_input("card5", min_value=0, max_value=300, value=226)
        addr1 = st.number_input("addr1 (Billing Zip)", min_value=0, max_value=500, value=299)
        addr2 = st.number_input("addr2", min_value=0, max_value=100, value=87)

    with col3:
        st.markdown("**📐 Behavioral Features**")
        c1    = st.number_input("C1 (Count)", min_value=0, max_value=2000, value=1)
        c2    = st.number_input("C2", min_value=0, max_value=5000, value=1)
        d1    = st.number_input("D1 (Days since first tx)", min_value=0.0, max_value=600.0, value=14.0)
        d10   = st.number_input("D10", min_value=0.0, max_value=600.0, value=10.0)
        tx_dt = st.number_input("TransactionDT (seconds)", min_value=0, value=86400 * 10)
        device = st.selectbox("DeviceType", ['desktop', 'mobile'])

    if st.button("⚡ Run Prediction", use_container_width=True, type="primary"):
        tx_hour   = int((tx_dt / 3600) % 24)
        input_row = {
            'TransactionID': 1,
            'TransactionDT': tx_dt,
            'TransactionAmt': tx_amt,
            'ProductCD': product_cd,
            'card1': card1, 'card2': card2, 'card3': card3, 'card5': card5,
            'addr1': addr1, 'addr2': addr2, 'dist1': np.nan,
            'C1': c1, 'C2': c2, 'C6': np.nan, 'C11': np.nan, 'C13': np.nan, 'C14': np.nan,
            'D1': d1, 'D2': np.nan, 'D3': np.nan, 'D4': np.nan, 'D10': d10, 'D15': np.nan,
            'card4': card4, 'card6': card6,
            'P_emaildomain': p_email, 'R_emaildomain': np.nan,
            'M4': np.nan, 'M6': np.nan,
            'id_30': np.nan, 'id_31': np.nan, 'DeviceType': device,
        }
        # Thêm V-cols mà model cần (sẽ là NaN nếu không có input)
        v_cols = [
            1,3,4,6,8,11,13,14,17,20,23,26,27,30,
            36,37,40,41,44,47,48,54,56,59,62,65,67,68,70,
            76,78,80,82,86,88,89,91,107,108,111,115,117,120,121,123,
            124,127,129,130,136,281,283,284,285,286,289,291,294,296,
            297,301,303,305,307,309,310,314,320,
        ]
        for v in v_cols:
            input_row[f'V{v}'] = np.nan

        df_input = pd.DataFrame([input_row])

        booster   = st.session_state.get('booster')
        uid_agg   = st.session_state.get('uid_agg')
        freq_maps = st.session_state.get('freq_maps')

        with st.spinner("Running feature engineering pipeline..."):
            try:
                if booster is not None and uid_agg is not None and freq_maps is not None:
                    X          = run_full_pipeline(df_input, uid_agg, freq_maps, booster)
                    prob       = float(predict(booster, X)[0])
                    mode_label = "XGBoost Model"
                else:
                    prob       = demo_predict({**input_row, 'P_emaildomain': p_email, 'tx_hour': tx_hour})
                    mode_label = "Demo Mode (heuristic)"
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback; st.code(traceback.format_exc())
                prob       = demo_predict({**input_row, 'P_emaildomain': p_email, 'tx_hour': tx_hour})
                mode_label = "Demo Mode (fallback)"

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

        with res_col1:
            st.metric("Fraud Probability", f"{prob*100:.1f}%")
            st.metric("Mode", mode_label)

        with res_col2:
            if prob >= threshold:
                st.markdown('<div class="fraud-badge">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
            elif prob >= threshold * 0.65:
                st.markdown('<div class="warn-badge">⚠ SUSPICIOUS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="legit-badge">✅ LEGITIMATE</div>', unsafe_allow_html=True)
            st.markdown(f"**Fraud threshold:** `{threshold}`")
            st.progress(prob)

        with res_col3:
            st.metric("tx_hour", f"{tx_hour}h")
            st.metric("log(Amt+1)", f"{np.log1p(tx_amt):.3f}")

        st.markdown("**🔍 Risk Signal Summary**")
        flags = []
        if _EMAIL_RISK.get(p_email, 0.18) > 0.3:
            flags.append(f"⚠ Email domain `{p_email}` — high-risk")
        if product_cd == 'C':
            flags.append("⚠ ProductCD=C — highest fraud rate (~14%)")
        if tx_amt < 5:
            flags.append(f"⚠ Micro-transaction ${tx_amt:.2f} — possible card testing")
        if tx_hour <= 5:
            flags.append(f"⚠ Night-time transaction ({tx_hour}h)")
        if c1 > 15:
            flags.append(f"⚠ C1={c1} — high frequency count")
        if not flags:
            flags.append("✅ No significant risk signals detected")
        for f in flags:
            st.markdown(f"- {f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — BATCH SCORING
# ═══════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("## 📋 Batch Scoring")
    st.markdown(
        "Upload file CSV (cùng format với `test_transaction.csv` của IEEE-CIS). "
        "App sẽ chạy **đúng** pipeline như notebook và trả về fraud probability."
    )

    uploaded = st.file_uploader(
        "Upload transaction CSV", type=["csv"],
        help="Cần có ít nhất: TransactionID, TransactionAmt, card1, addr1, D1, TransactionDT"
    )

    if uploaded:
        df_up = pd.read_csv(uploaded, nrows=5000)
        st.success(f"Loaded {len(df_up):,} rows × {df_up.shape[1]} cols")

        with st.expander("Preview data"):
            st.dataframe(df_up.head(10))

        if st.button("Score Batch", use_container_width=True, type="primary"):
            booster   = st.session_state.get('booster')
            uid_agg   = st.session_state.get('uid_agg')
            freq_maps = st.session_state.get('freq_maps')

            with st.spinner(f"Scoring {len(df_up):,} transactions..."):
                try:
                    if booster is not None and uid_agg is not None and freq_maps is not None:
                        # ── BUG FIX: dùng run_full_pipeline mới lấy features từ booster ──
                        X_batch = run_full_pipeline(df_up.copy(), uid_agg, freq_maps, booster)
                        probs   = predict(booster, X_batch)

                    else:
                        # Demo mode
                        prog  = st.progress(0)
                        probs = []
                        for i, (_, row) in enumerate(df_up.iterrows()):
                            probs.append(demo_predict(row.to_dict()))
                            if i % max(1, len(df_up) // 20) == 0:
                                prog.progress(min(i / len(df_up), 1.0))
                        probs = np.array(probs)
                        st.info("⚠ Demo mode — load artifacts để dùng model thật")

                    id_col    = 'TransactionID' if 'TransactionID' in df_up.columns else df_up.columns[0]
                    df_result = df_up[[id_col]].copy()
                    df_result['fraud_probability'] = np.round(probs, 4)
                    df_result['prediction']        = (probs >= threshold).astype(int)
                    df_result['label']             = df_result['prediction'].map({0: 'Legit', 1: 'Fraud'})

                    n_fraud = int((probs >= threshold).sum())
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Transactions", f"{len(probs):,}")
                    m2.metric("Fraud Detected",     f"{n_fraud:,}")
                    m3.metric("Legitimate",          f"{len(probs)-n_fraud:,}")
                    m4.metric("Fraud Rate",          f"{n_fraud/len(probs)*100:.2f}%")

                    st.dataframe(
                        df_result.sort_values('fraud_probability', ascending=False).head(50),
                        use_container_width=True,
                    )

                    csv_out = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇ Download Predictions CSV",
                        data=csv_out,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    fig = px.histogram(
                        df_result, x='fraud_probability', color='label',
                        color_discrete_map={'Legit': '#3fb950', 'Fraud': '#f85149'},
                        nbins=50, barmode='overlay', opacity=0.7,
                        title='Fraud Probability Distribution',
                    )
                    fig.update_layout(
                        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
                        font_color='#c9d1d9',
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Scoring error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — EDA (giữ nguyên, không liên quan bug)
# ═══════════════════════════════════════════════════════════════════
def plot_label_dist(df: pd.DataFrame) -> go.Figure:
    counts = df['isFraud'].value_counts().reset_index()
    counts.columns = ['isFraud', 'count']
    counts['label'] = counts['isFraud'].map({0: 'Legit', 1: 'Fraud'})
    fig = px.pie(
        counts, values='count', names='label',
        color='label',
        color_discrete_map={'Legit': '#3fb950', 'Fraud': '#f85149'},
        hole=0.45, title='Label Distribution',
    )
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                      font_color='#c9d1d9', title_font_size=14,
                      legend=dict(font_color='#c9d1d9'))
    return fig


def plot_amt_distribution(df: pd.DataFrame) -> go.Figure:
    sample = df[['TransactionAmt', 'isFraud']].dropna().sample(
        min(len(df), 20000), random_state=42)
    sample['log_amt'] = np.log1p(sample['TransactionAmt'])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['log(TransactionAmt+1) by Label', 'Boxplot by Label'])
    colors = {0: '#3fb950', 1: '#f85149'}
    labels = {0: 'Legit', 1: 'Fraud'}
    for label in [0, 1]:
        sub = sample[sample['isFraud'] == label]['log_amt']
        fig.add_trace(go.Histogram(x=sub, name=labels[label], marker_color=colors[label],
                                   opacity=0.6, nbinsx=60, histnorm='probability density'),
                      row=1, col=1)
        fig.add_trace(go.Box(y=sub, name=labels[label], marker_color=colors[label],
                             boxmean=True), row=1, col=2)
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
                      font_color='#c9d1d9', title_text='Transaction Amount Analysis',
                      barmode='overlay', height=380)
    return fig


def plot_fraud_rate_by_cat(df: pd.DataFrame, col: str) -> go.Figure:
    agg = (df.groupby(col)['isFraud'].agg(['mean', 'count']).reset_index()
           .rename(columns={'mean': 'fraud_rate', 'count': 'total'}))
    agg['fraud_rate_pct'] = (agg['fraud_rate'] * 100).round(2)
    agg = agg.sort_values('fraud_rate_pct', ascending=False).head(15)
    fig = px.bar(agg, x=col, y='fraud_rate_pct', text='fraud_rate_pct',
                 color='fraud_rate_pct',
                 color_continuous_scale=['#3fb950', '#d2993a', '#f85149'],
                 title=f'Fraud Rate (%) by {col}',
                 labels={'fraud_rate_pct': 'Fraud Rate (%)'})
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
                      font_color='#c9d1d9', coloraxis_showscale=False,
                      height=380, title_font_size=14)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols  = ['TransactionAmt', 'C1', 'C2', 'C6', 'C11', 'C13',
                 'D1', 'D4', 'D10', 'card1', 'addr1', 'isFraud']
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr(numeric_only=True)
    fig  = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                                y=corr.index.tolist(), colorscale='RdBu', zmid=0,
                                text=np.round(corr.values, 2), texttemplate='%{text}',
                                textfont_size=9))
    fig.update_layout(title='Feature Correlation Matrix', paper_bgcolor='#0d1117',
                      plot_bgcolor='#0d1117', font_color='#c9d1d9', height=420)
    return fig


def plot_hourly_fraud(df: pd.DataFrame) -> go.Figure:
    if 'tx_hour' not in df.columns:
        df = df.copy()
        df['tx_hour'] = ((df.get('TransactionDT', 0) / 3600) % 24).astype(int)
    agg = df.groupby('tx_hour')['isFraud'].agg(['mean', 'count']).reset_index()
    agg['fraud_rate_pct'] = (agg['mean'] * 100).round(2)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg['tx_hour'], y=agg['count'], name='Total Transactions',
                         marker_color='#1f6feb', opacity=0.5), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg['tx_hour'], y=agg['fraud_rate_pct'],
                             name='Fraud Rate (%)', mode='lines+markers',
                             line=dict(color='#f85149', width=2), marker=dict(size=6)),
                  secondary_y=True)
    fig.update_layout(title='Fraud Rate & Volume by Hour of Day', paper_bgcolor='#0d1117',
                      plot_bgcolor='#161b22', font_color='#c9d1d9', height=360)
    fig.update_yaxes(title_text='Transaction Count', secondary_y=False)
    fig.update_yaxes(title_text='Fraud Rate (%)', secondary_y=True)
    return fig


with tab_eda:
    st.markdown("## Exploratory Data Analysis")

    eda_uploaded = st.file_uploader(
        "Upload train_transaction.csv (optional)", type=["csv"], key="eda_upload")

    @st.cache_data
    def generate_sample_eda_data(n=8000) -> pd.DataFrame:
        np.random.seed(42)
        n_fraud = int(n * 0.035)
        n_legit = n - n_fraud
        fraud = pd.DataFrame({
            'isFraud': 1,
            'TransactionAmt': np.concatenate([
                np.random.lognormal(4.0, 0.4, n_fraud // 2),
                np.random.lognormal(5.3, 0.3, n_fraud - n_fraud // 2),
            ]),
            'ProductCD':    np.random.choice(['C','H','R','W','S'], p=[0.38,0.22,0.16,0.14,0.10], size=n_fraud),
            'card4':        np.random.choice(['visa','mastercard','discover','american express'], p=[0.45,0.30,0.15,0.10], size=n_fraud),
            'card6':        np.random.choice(['debit','credit','charge card'], p=[0.55,0.38,0.07], size=n_fraud),
            'P_emaildomain':np.random.choice(['gmail.com','yahoo.com','anonymous.com','hotmail.com','protonmail.com','unknown.xyz'], p=[0.25,0.20,0.22,0.15,0.10,0.08], size=n_fraud),
            'DeviceType':   np.random.choice(['desktop','mobile'], p=[0.48,0.52], size=n_fraud),
            'C1':  np.random.exponential(12, n_fraud).astype(int),
            'D1':  np.random.uniform(0, 300, n_fraud),
            'D4':  np.random.uniform(0, 500, n_fraud),
            'D10': np.random.uniform(0, 500, n_fraud),
            'card1': np.random.randint(1000, 18000, n_fraud),
            'addr1': np.random.randint(10, 490, n_fraud),
            'TransactionDT': np.random.randint(0, 86400*180, n_fraud),
        })
        legit = pd.DataFrame({
            'isFraud': 0,
            'TransactionAmt': np.random.lognormal(4.3, 1.1, n_legit),
            'ProductCD':    np.random.choice(['W','H','C','S','R'], p=[0.55,0.18,0.12,0.10,0.05], size=n_legit),
            'card4':        np.random.choice(['visa','mastercard','discover','american express'], p=[0.60,0.28,0.07,0.05], size=n_legit),
            'card6':        np.random.choice(['debit','credit','charge card'], p=[0.62,0.35,0.03], size=n_legit),
            'P_emaildomain':np.random.choice(['gmail.com','yahoo.com','anonymous.com','hotmail.com','protonmail.com','unknown.xyz'], p=[0.42,0.28,0.06,0.18,0.04,0.02], size=n_legit),
            'DeviceType':   np.random.choice(['desktop','mobile'], p=[0.58,0.42], size=n_legit),
            'C1':  np.random.exponential(3, n_legit).astype(int),
            'D1':  np.random.uniform(0, 600, n_legit),
            'D4':  np.random.uniform(0, 600, n_legit),
            'D10': np.random.uniform(0, 600, n_legit),
            'card1': np.random.randint(1000, 18000, n_legit),
            'addr1': np.random.randint(10, 490, n_legit),
            'TransactionDT': np.random.randint(0, 86400*180, n_legit),
        })
        df = pd.concat([fraud, legit], ignore_index=True).sample(frac=1, random_state=42)
        df['tx_hour'] = ((df['TransactionDT'] / 3600) % 24).astype(int)
        df['C2']  = np.random.exponential(3, len(df)).astype(int)
        df['C6']  = np.random.exponential(2, len(df)).astype(int)
        df['C11'] = np.random.exponential(4, len(df)).astype(int)
        df['C13'] = np.random.exponential(5, len(df)).astype(int)
        return df

    df_eda = pd.read_csv(eda_uploaded, nrows=50000) if eda_uploaded else generate_sample_eda_data()
    if not eda_uploaded:
        st.info("ℹ Using synthetic sample data (upload train CSV for real EDA)")

    st.markdown("### 1️⃣ Label Distribution")
    c1, c2 = st.columns([1, 2])
    with c1:
        n_f = int(df_eda['isFraud'].sum()); n_t = len(df_eda)
        st.metric("Total", f"{n_t:,}")
        st.metric("🚨 Fraud", f"{n_f:,} ({n_f/n_t*100:.2f}%)")
        st.metric("✅ Legit", f"{n_t-n_f:,} ({(n_t-n_f)/n_t*100:.2f}%)")
    with c2:
        st.plotly_chart(plot_label_dist(df_eda), use_container_width=True)

    st.markdown("---")
    st.markdown("### 2️⃣ Transaction Amount Analysis")
    st.plotly_chart(plot_amt_distribution(df_eda), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3️⃣ Fraud Rate by Categorical Features")
    cat_choice = st.selectbox("Chọn feature",
        [c for c in ['ProductCD','card4','card6','P_emaildomain','DeviceType'] if c in df_eda.columns])
    st.plotly_chart(plot_fraud_rate_by_cat(df_eda, cat_choice), use_container_width=True)

    st.markdown("---")
    st.markdown("### 4️⃣ Fraud Rate by Hour of Day")
    st.plotly_chart(plot_hourly_fraud(df_eda), use_container_width=True)

    st.markdown("---")
    st.markdown("### 5️⃣ Feature Correlation Heatmap")
    st.plotly_chart(plot_correlation_heatmap(df_eda), use_container_width=True)

    st.markdown("---")
    st.markdown("### 6️⃣ C1 Count Feature Distribution")
    if 'C1' in df_eda.columns:
        fig_c1 = px.histogram(
            df_eda[df_eda['C1'] < 50], x='C1',
            color=df_eda[df_eda['C1'] < 50]['isFraud'].astype(str),
            barmode='overlay', opacity=0.65, nbins=40,
            color_discrete_map={'0': '#3fb950', '1': '#f85149'},
            title='C1 (Transaction Count) Distribution by Label',
            labels={'color': 'isFraud'},
        )
        fig_c1.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#161b22', font_color='#c9d1d9')
        st.plotly_chart(fig_c1, use_container_width=True)