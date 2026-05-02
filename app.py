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
from datetime import datetime
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import pickle
import os
import yaml
from db import save_case, get_all_cases, update_case_status, get_case_by_id, get_related_cases
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── INITIALIZE GLOBAL STATE ────────────────────────────────────────────────
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = 0.5

# ─── INITIALIZE GLOBAL STATE ────────────────────────────────────────────────
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = 0.5

# ─── LOAD AUTHENTICATION CONFIG ──────────────────────────────────────────────
with open('auth_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    auto_hash=True # Allows you to type plain text password directly in YAML
)

try:
    authenticator.login(
        location='main',
        fields={
            'Form name': 'Sentinel Intel - Fraud Detection', 
            'Username': 'Username', 
            'Password': 'Password', 
            'Login': 'Login'
        }
    )
except Exception as e:
    st.error(f"Auth System Error: {e}")

# ─── LOGIC CHECK LOGIN ──────────────────────────────────────────────────────
if st.session_state["authentication_status"] is False:
    st.error('Incorrect username or password. Please try again.')
    st.stop()
elif st.session_state["authentication_status"] is None:
    # Không hiện warning gây khó chịu, chỉ hiện thông tin hướng dẫn
    st.info('Please login to access the Fraud Detection system.')
    st.stop()


# ─── THEME & STYLE SETTINGS ──────────────────────────────────────────────────
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
FEATURE_COLS_ORDER = [
    'TransactionID', 'TransactionDT', 'TransactionAmt',
    'card1', 'card2', 'card3', 'card5',
    'addr1', 'addr2', 'dist1',
    'C1', 'C2', 'C6', 'C11', 'C13', 'C14',
    'D1', 'D10', 'D15',
    'ProductCD_freq', 'card4_freq', 'card6_freq',
    'P_emaildomain_freq', 'R_emaildomain_freq',
    'M4_freq', 'M6_freq', 'id_30_freq', 'id_31_freq', 'DeviceType_freq',
    # Thêm các V-cols:
    'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17', 'V20', 'V23', 'V26', 'V27', 'V30',
    'V36', 'V37', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V56', 'V59', 'V62', 'V65', 'V67', 'V68', 'V70',
    'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89', 'V91', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121', 'V123',
    'V124', 'V127', 'V129', 'V130', 'V136', 'V281', 'V283', 'V284', 'V285', 'V286', 'V289', 'V291', 'V294', 'V296',
    'V297', 'V301', 'V303', 'V305', 'V307', 'V309', 'V310', 'V314', 'V320'
]
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
    # BUG FIX: drop cột trùng trước khi merge và xử lý fillna thông minh
    uid_cols = [c for c in uid_agg.columns if c != 'uid']
    df = df.drop(columns=[c for c in uid_cols if c in df.columns], errors='ignore')
    df = df.merge(uid_agg, on='uid', how='left')
    
    # KIỂM TRA TOÀN VẸN: Chỉ fillna nếu cột tồn tại trong database nạp vào
    # Giúp ứng dụng không bị crash khi dùng các model/data schema khác nhau
    if 'uid_amt_mean' in df.columns:
        df['uid_amt_mean'] = df['uid_amt_mean'].fillna(df['TransactionAmt'])
    
    if 'uid_amt_std' in df.columns:
        df['uid_amt_std'] = df['uid_amt_std'].fillna(20.0)
    
    if 'uid_C1_mean' in df.columns:
        df['uid_C1_mean'] = df['uid_C1_mean'].fillna(1.0)
        
    if 'uid_D1_mean' in df.columns:
        df['uid_D1_mean'] = df['uid_D1_mean'].fillna(10.0)
        
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

        # Khôi phục NaN cho categorical gốc - Sửa lỗi FutureWarning
        df[col_name] = df[col_name].replace({'nan': np.nan, 'None': np.nan, '': np.nan}).infer_objects(copy=False)
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
    df = add_uid(df)
    df = apply_uid_agg(df, uid_agg)
    df = apply_freq_encoding(df, freq_maps)
    df = add_extra_features(df)

    # Đảm bảo đồng bộ đặc trưng
    features = booster.feature_names if booster.feature_names else FEATURE_COLS_ORDER
    
    # Xử lý các cột Categorical còn sót lại (chưa được freq encoded)
    for col in features:
        if col not in df.columns:
            df[col] = np.nan

        # Nếu cột vẫn là dạng chuỗi (Object), mã hóa nó thành số để model hiểu
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.factorize(df[col].astype(str))[0].astype(np.float32)
  
    # Trích xuất và ép kiểu an toàn
    X = df[features].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    return X


def predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Thực hiện dự báo an toàn"""
    if booster.feature_names is not None:
        dmat = xgb.DMatrix(X[booster.feature_names])
    else:
        dmat = xgb.DMatrix(X.values)
    return booster.predict(dmat)


def update_uid_stats(new_rows_df: pd.DataFrame, uid_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Cập nhật dữ liệu tổng hợp (uid_agg) từ giao dịch mới và lưu lại bản 'live'.
    """
    uid_agg = uid_agg.copy()
    live_path = "./artifacts/uid_agg_live.parquet"
    
    # Đảm bảo có cột uid để tra cứu
    if 'uid' not in new_rows_df.columns:
        new_rows_df = add_uid(new_rows_df)
        
    for _, row in new_rows_df.iterrows():
        uid = str(row['uid'])
        amt = float(row.get('TransactionAmt', 0))
        d10 = float(row.get('D10', 0)) if not pd.isna(row.get('D10')) else 0.0
        c1  = float(row.get('C1', 0)) if not pd.isna(row.get('C1')) else 0.0
        
        if uid in uid_agg['uid'].values:
            idx = uid_agg.index[uid_agg['uid'] == uid][0]
            n = uid_agg.at[idx, 'uid_tx_count']
            
            # Cập nhật các chỉ số trung bình (Online Mean)
            uid_agg.at[idx, 'uid_amt_mean'] = (uid_agg.at[idx, 'uid_amt_mean'] * n + amt) / (n + 1)
            uid_agg.at[idx, 'uid_D10_mean'] = (uid_agg.at[idx, 'uid_D10_mean'] * n + d10) / (n + 1)
            uid_agg.at[idx, 'uid_C1_mean']  = (uid_agg.at[idx, 'uid_C1_mean'] * n + c1) / (n + 1)
            
            # Cập nhật Min/Max
            uid_agg.at[idx, 'uid_amt_max'] = max(uid_agg.at[idx, 'uid_amt_max'], amt)
            uid_agg.at[idx, 'uid_amt_min'] = min(uid_agg.at[idx, 'uid_amt_min'], amt)
            
            # Tăng số lượng giao dịch
            uid_agg.at[idx, 'uid_tx_count'] = n + 1
        else:
            # Nếu là khách hàng hoàn toàn mới
            new_record = {
                'uid': uid, 'uid_tx_count': 1,
                'uid_amt_mean': amt, 'uid_amt_std': 0.0,
                'uid_amt_max': amt, 'uid_amt_min': amt,
                'uid_D10_mean': d10, 'uid_C1_mean': c1
            }
            uid_agg = pd.concat([uid_agg, pd.DataFrame([new_record])], ignore_index=True)
    
    # Lưu xuống file để duy trì trạng thái sau khi tắt app
    try:
        if not os.path.exists("./artifacts"):
            os.makedirs("./artifacts")
        uid_agg.to_parquet(live_path)
    except Exception as e:
        print(f"Error saving live UID stats: {e}")
        
    return uid_agg



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


# ─── SIDEBAR NAVIGATION & ASSET MANAGEMENT ──────────────────────────────────
with st.sidebar:    
    # Navigation
    page = option_menu(
        menu_title=None, 
        options=["Dashboard", "Predict", "Investigation Center", "Batch Scoring", "EDA Analytics", "Settings"],
        icons=["speedometer2", "lightning-charge", "shield-check", "layers-half", "bar-chart-steps", "gear"],
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#161b22"},
            "icon": {"color": "#58a6ff", "font-size": "18px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#21262d"},
            "nav-link-selected": {"background-color": "#238636"},
        }
    )

    st.write("---")
    # 🚪 Official Logout Widget
    authenticator.logout('Logout', 'sidebar')
    st.write("---")

# ─── MAIN CONTENT ROUTING ──────────────────────────────────────────────────

if page == "Dashboard":
    st.markdown("# Executive Fraud Dashboard")
    st.markdown("Real-time risk insights from aggregated transaction history.")
    
    # ─── LIVE PRODUCTION METRICS (from sentinel.db) ──────────────────────────
    st.subheader("🚀 Live Production Monitoring")
    df_cases = get_all_cases()
    threshold = st.session_state.get('threshold', 0.5)
    
    if not df_cases.empty:
        # Calculate live metrics
        total_cases = len(df_cases)
        high_risk_cases = len(df_cases[df_cases['probability'] >= threshold])
        confirmed_fraud = len(df_cases[df_cases['status'] == 'Confirmed Fraud'])
        pending_cases = len(df_cases[df_cases['status'] == 'Pending'])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Scanned", f"{total_cases:,}")
        m2.metric("High Risk Detected", f"{high_risk_cases:,}", 
                  delta=f"{high_risk_cases/total_cases*100:.1f}% Risk Rate", delta_color="inverse")
        
        # Calculate accuracy/precision if there are confirmed labels
        confirmed_total = len(df_cases[df_cases['status'].isin(['Confirmed Fraud', 'Confirmed Legit'])])
        if confirmed_total > 0:
            precision = (confirmed_fraud / confirmed_total * 100)
            m3.metric("Confirmed Fraud", f"{confirmed_fraud:,}", delta=f"{precision:.1f}% Prec.")
        else:
            m3.metric("Confirmed Fraud", f"{confirmed_fraud:,}", delta="No Labels Yet")
            
        m4.metric("Pending Review", f"{pending_cases:,}")
        
        col_l, col_r = st.columns(2)
        with col_l:
            # Probability Distribution
            fig_prob = px.histogram(
                df_cases, x='probability', 
                title="Risk Probability Distribution (Live)",
                template="plotly_dark", 
                color_discrete_sequence=['#f85149'],
                nbins=30
            )
            fig_prob.add_vline(x=threshold, line_dash="dash", line_color="white", annotation_text="Threshold")
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with col_r:
            # Status Breakdown
            status_counts = df_cases['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_status = px.pie(
                status_counts, values='Count', names='Status',
                title="Investigation Status Breakdown",
                template="plotly_dark",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_status, use_container_width=True)
            
        # Top Risk Entities
        st.markdown("### 🔍 Top High-Risk Entities")
        high_risk_df = df_cases[df_cases['probability'] >= threshold]
        if not high_risk_df.empty:
            c_entity1, c_entity2 = st.columns(2)
            with c_entity1:
                risk_cards = high_risk_df.groupby('card1').size().reset_index(name='fraud_count')
                risk_cards = risk_cards.sort_values('fraud_count', ascending=False).head(10)
                fig_cards = px.bar(
                    risk_cards, x='card1', y='fraud_count',
                    title="Top 10 High-Risk Card1 IDs",
                    template="plotly_dark",
                    color='fraud_count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_cards, use_container_width=True)
            
            with c_entity2:
                risk_emails = high_risk_df.groupby('email').size().reset_index(name='fraud_count')
                risk_emails = risk_emails.sort_values('fraud_count', ascending=False).head(10)
                risk_emails['email'] = risk_emails['email'].replace('', 'Unknown')
                fig_emails = px.bar(
                    risk_emails, x='email', y='fraud_count',
                    title="Top 10 High-Risk Emails",
                    template="plotly_dark",
                    color='fraud_count',
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig_emails, use_container_width=True)
    else:
        st.info("No live prediction data found in sentinel.db. Statistics will appear here after you run predictions.")

    st.write("---")
    
    # ─── HISTORICAL BASELINE (from parquet) ──────────────────────────────────
    st.subheader("📊 Historical Baseline (Training Set)")
    uid_df = st.session_state.get('uid_agg')
    if uid_df is None:
        live_p = "./artifacts/uid_agg_live.parquet"
        sample_p = "./artifacts/uid_agg_sample.parquet"
        if os.path.exists(live_p):
            uid_df = load_uid_agg(live_p)
        elif os.path.exists(sample_p):
            uid_df = load_uid_agg(sample_p)
        st.session_state['uid_agg'] = uid_df

    if uid_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Transactions", f"{uid_df['uid_tx_count'].sum():,}")
        c2.metric("Unique Entities", f"{len(uid_df):,}")
        c3.metric("Avg Training Amt", f"${uid_df['uid_amt_mean'].mean():.2f}")
        c4.metric("Baseline Risk Users", f"{len(uid_df[uid_df['uid_amt_mean'] > 500]):,}")
        
        col_left, col_right = st.columns(2)
        with col_left:
            fig_vol = px.histogram(uid_df, x='uid_tx_count', title="Training Activity Distribution (Volume)", 
                                   template="plotly_dark", color_discrete_sequence=['#2563eb'])
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with col_right:
            fig_amt = px.box(uid_df, y='uid_amt_mean', title="Training Amt Spread per UID", 
                             template="plotly_dark", color_discrete_sequence=['#f85149'])
            st.plotly_chart(fig_amt, use_container_width=True)
    else:
        st.warning("Historical baseline artifacts not found.")


elif page == "Investigation Center":
    st.markdown("# 🛡️ Investigation Center")
    st.markdown("Manage and update the status of suspicious fraud cases.")

    df_cases = get_all_cases()
    
    if df_cases.empty:
        st.info("No files have been recorded yet. Please run the prediction on the 'Predict' page to start.")
    else:
        # Statistics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cases", len(df_cases))
        c2.metric("❌ Fraud", len(df_cases[df_cases['status'] == 'Confirmed Fraud']))
        c3.metric("✅ Legit", len(df_cases[df_cases['status'] == 'Confirmed Legit']))
        c4.metric("⏳ Pending", len(df_cases[df_cases['status'] == 'Pending']))
        st.write("---")
        
        # Dashboard Table with Status Update
        st.markdown("### List of files")
        
        status_filter = st.multiselect(
            "Filter by status", 
            ['Pending', 'Confirmed Fraud', 'Confirmed Legit', 'In Review'], 
            default=['Pending', 'In Review']
        )
        
        filtered_df = df_cases[df_cases['status'].isin(status_filter)]
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "id": "#ID",
                "probability": st.column_config.ProgressColumn("Xác suất Fraud", min_value=0, max_value=1),
                "created_at": "Thời gian tạo",
                "status": "Trạng thái",
            }
        )

        # 🚀 EXPORT DASHBOARD AS REPORT
        st.write("")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Investigation Report (CSV)",
            data=csv_data,
            file_name=f"sentinel_investigation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
            use_container_width=True,
            type="secondary",
            help="Xuất danh sách các hồ sơ đang hiển thị trong bộ lọc hiện tại."
        )

        st.write("---")
        st.markdown("### Update profile status")
        with st.form("update_case_form"):
            case_id_to_up = st.number_input("Select the profile ID that needs updating.", min_value=1, step=1)
            new_status = st.selectbox("New status", ['Confirmed Fraud', 'Confirmed Legit', 'In Review', 'Resolved'])
            new_notes = st.text_area("Investigation notes")
            
            if st.form_submit_button("Update profile"):
                if case_id_to_up in df_cases['id'].values:
                    update_case_status(case_id_to_up, new_status, new_notes)
                    st.success(f"Updated profile #{case_id_to_up} to {new_status}!")
                    st.rerun()
                else:
                    st.error("Profile ID not found.")
elif page == "Predict":
    st.markdown("## Single Transaction Scoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💳 Transaction Info**")
        tx_amt     = st.number_input("TransactionAmt ($)", min_value=0.01, value=125.50, step=0.01)
        product_cd = st.selectbox("ProductCD", ['W', 'H', 'C', 'S', 'R'])
        card4      = st.selectbox("Card Network", ['visa', 'mastercard', 'discover', 'american express'])
        card6      = st.selectbox("Card Type", ['debit', 'credit', 'charge card'])
        p_email    = st.selectbox("P_emaildomain", [
            'gmail.com', 'yahoo.com', 'hotmail.com',
            'anonymous.com', 'protonmail.com', 'unknown.xyz', 'other'
        ])

    with col2:
        st.markdown("**🏦 Card & Address**")
        card1 = st.number_input("Bank ID", min_value=0, max_value=20000, value=9500)
        card2 = st.number_input("Branch ID", min_value=0, max_value=1000, value=321)
        card3 = st.number_input("Country ID", min_value=0, max_value=200, value=150)
        card5 = st.number_input("Category ID", min_value=0, max_value=300, value=226)
        addr1 = st.number_input("Billing Zip", min_value=0, max_value=500, value=299)
        addr2 = st.number_input("Billing Country", min_value=0, max_value=100, value=87)

    with col3:
        st.markdown("**📐 Behavioral Features**")
        c1    = st.number_input("Address/Email Count", min_value=0, max_value=2000, value=1)
        c2    = st.number_input("Card Usage Count", min_value=0, max_value=5000, value=1)
        d1    = st.number_input("Account Age (Days)", min_value=0.0, max_value=600.0, value=14.0)
        d10   = st.number_input("Days Since Last Activity", min_value=0.0, max_value=600.0, value=10.0)
        tx_dt = st.number_input("TransactionDT (seconds)", min_value=0, value=86400 * 10)
        device = st.selectbox("DeviceType", ['desktop', 'mobile'])
    
    threshold = st.session_state['threshold']
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
                    
                    # ── FEEDBACK LOOP: Cập nhật dữ liệu lịch sử ngay lập tức ──
                    st.session_state['uid_agg'] = update_uid_stats(df_input, uid_agg)
                    
                else:
                    prob       = demo_predict({**input_row, 'P_emaildomain': p_email, 'tx_hour': tx_hour})
                    mode_label = "Demo Mode (heuristic)"
                
                # ── AUTOMATIC SAVE TO DATABASE (Log all transactions) ──
                txn_id = f"SING-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                investigator = st.session_state.get("username", "System")
                is_fraud = prob >= threshold
                label = "Fraud" if is_fraud else "Legit"
                initial_status = "Pending" if is_fraud else "Confirmed Legit"
                
                saved = save_case(
                    transaction_id=txn_id,
                    probability=prob,
                    prediction_label=label,
                    card1=card1,
                    addr1=addr1,
                    email=p_email,
                    investigator=investigator,
                    status=initial_status
                )
                if saved:
                    if is_fraud:
                        st.toast(f"🚨 Fraud case logged: {txn_id}", icon="🛡️")
                    else:
                        st.toast(f"✅ Transaction logged: {txn_id}", icon="📊")
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


elif page == "Batch Scoring":
    st.markdown("## Batch Scoring")
    st.markdown(
        "Diagnose fraud risk by uploading a CSV file. The application performs feature analysis and returns a real-time risk probability index."
    )

    uploaded = st.file_uploader(
        "Upload transaction CSV", type=["csv"],
        help="Must contain at least: TransactionID, TransactionAmt, card1, addr1, D1, TransactionDT"
    )

    if uploaded:
        df_up = pd.read_csv(uploaded, nrows=5000)
        st.success(f"Loaded {len(df_up):,} rows × {df_up.shape[1]} cols")

        with st.expander("Preview data"):
            st.dataframe(df_up.head(10))

        if st.button("🚀 Score Batch", use_container_width=True, type="primary"):
            booster   = st.session_state.get('booster')
            uid_agg   = st.session_state.get('uid_agg')
            freq_maps = st.session_state.get('freq_maps')

            with st.spinner(f"Scoring {len(df_up):,} transactions..."):
                try:
                    if booster is not None and uid_agg is not None and freq_maps is not None:
                        # ── BUG FIX: dùng run_full_pipeline mới lấy features từ booster ──
                        X_batch = run_full_pipeline(df_up.copy(), uid_agg, freq_maps, booster)
                        probs   = predict(booster, X_batch)
                        
                        # ── FEEDBACK LOOP: Cập nhật lịch sử cho toàn bộ lô giao dịch ──
                        st.session_state['uid_agg'] = update_uid_stats(df_up, uid_agg)

                    else:
                        # Demo mode
                        prog  = st.progress(0)
                        probs = []
                        for i, (_, row) in enumerate(df_up.iterrows()):
                            probs.append(demo_predict(row.to_dict()))
                            if i % max(1, len(df_up) // 20) == 0:
                                prog.progress(min(i / len(df_up), 1.0))
                        probs = np.array(probs)
                        st.info("⚠ Demo mode — load artifacts to use the real model")

                    id_col    = 'TransactionID' if 'TransactionID' in df_up.columns else df_up.columns[0]
                    df_result = df_up[[id_col]].copy()
                    threshold = st.session_state['threshold']
                    df_result['fraud_probability'] = np.round(probs, 4)
                    df_result['prediction']        = (probs >= threshold).astype(int)
                    df_result['label']             = df_result['prediction'].map({0: 'Legit', 1: 'Fraud'})

                    # ── AUTOMATIC SAVE ALL BATCH TRANSACTIONS ──
                    investigator = st.session_state.get("username", "System")
                    new_logs_count = 0
                    fraud_count = 0
                    
                    for index, row in df_result.iterrows():
                        orig_row = df_up.iloc[index]
                        is_fraud = row['prediction'] == 1
                        initial_status = "Pending" if is_fraud else "Confirmed Legit"
                        
                        saved = save_case(
                            transaction_id=row[id_col],
                            probability=row['fraud_probability'],
                            prediction_label=row['label'],
                            card1=orig_row.get('card1', ''),
                            addr1=orig_row.get('addr1', ''),
                            email=orig_row.get('P_emaildomain', ''),
                            investigator=investigator,
                            status=initial_status
                        )
                        if saved:
                            new_logs_count += 1
                            if is_fraud: fraud_count += 1
                            
                    if new_logs_count > 0:
                        st.toast(f"✅ Processed {new_logs_count} transactions ({fraud_count} high risk)", icon="📊")


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

elif page == "EDA Analytics":
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
    cat_choice = st.selectbox("SELECT feature",
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

elif page == "Settings":
    st.markdown("### ⚙️ System Configuration")

    st.info("💡 You can upload a new file or leave it blank to use the default file set.")

    # File Uploaders
    up_model = st.file_uploader("Update Model (.ubj)", type=['ubj'])

    st.write("---")
    st.session_state['threshold'] = st.slider(
        "Fraud Threshold (Global)", 
        min_value=0.1, 
        max_value=0.9, 
        value=st.session_state['threshold'], 
        step=0.05,
        help="This threshold will be applied to all predictions in the system."
    )

    if st.button("🔄 Sync & Refresh Assets", use_container_width=True, type="primary"):
        # Default Hardcoded Paths
        DEF_M = "./best_xgb_model/best_booster_sample.ubj"
        DEF_U = "./artifacts/uid_agg_sample.parquet"
        DEF_F = "./artifacts/freq_maps_sample.pkl"

        # 1. Sync Model
        try:
            live_u = "./artifacts/uid_agg_live.parquet"
            u_path = live_u if os.path.exists(live_u) else DEF_U

            if up_model is not None:
                with open("temp_model.ubj", "wb") as f:
                    f.write(up_model.getbuffer())
                st.session_state['booster'] = load_model("temp_model.ubj")
                st.session_state['uid_agg'] = load_uid_agg(u_path)
                st.session_state['freq_maps'] = load_freq_maps(DEF_F)
                st.success("✅ New Model & Latest UID Stats Loaded!")
            else:
                st.session_state['booster'] = load_model(DEF_M)
                st.session_state['uid_agg'] = load_uid_agg(u_path)
                st.session_state['freq_maps'] = load_freq_maps(DEF_F)
                st.success(f"ℹ️ Default Assets Synced (Source: {'Live' if u_path == live_u else 'Sample'}).")
        except Exception as e: st.error(f"Model Error: {e}")

 