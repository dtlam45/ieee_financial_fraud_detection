# 🛡️ IEEE Financial Fraud Detection

Dự án phát hiện giao dịch gian lận tài chính dựa trên bộ dữ liệu **IEEE-CIS Fraud Detection** (Kaggle). Sử dụng **XGBoost** kết hợp **PySpark** để xử lý dữ liệu lớn, kèm giao diện demo trực quan bằng **Streamlit**.

---

## 📁 Cấu trúc project

```
ieee_financial_fraud_detection/
│
├── ieee_fraud_detection.ipynb   
├── predict_fraud.ipynb        
├── app.py                       # Streamlit demo app
│
├── artifacts/
│   ├── uid_agg.parquet          # Aggregated features theo UID
│   └── freq_maps.pkl            # Frequency encoding maps cho categorical features
│
├── best_xgb_model/
│   └── best_booster.ubj         # XGBoost model đã train
│
├── pyproject.toml               # Dependencies
└── .gitignore
```

---

## 🧠 Feature Engineering

- **UID feature**: Tạo định danh người dùng từ `card1 + addr1 + D1_norm`
- **UID aggregation**: Thống kê hành vi theo UID (mean/std/count của TransactionAmt)
- **Frequency encoding**: Mã hóa tần suất cho 10 categorical features
- **Extra features**: `tx_hour`, `tx_day_of_week`, `log_TransactionAmt`, `amt_to_uid_mean_ratio`
- **V-features**: 60+ Vesta engineered features

---

## 🚀 Cài đặt & Chạy

### 1. Clone repo
```bash
git clone https://github.com/dtlam45/ieee_financial_fraud_detection.git
cd ieee_financial_fraud_detection
```

### 2. Cài dependencies
```bash
uv sync
```

### 3. Tải data
Download từ Kaggle và đặt vào thư mục gốc:
- [`train_transaction.csv`](https://www.kaggle.com/c/ieee-fraud-detection/data)
- `test_transaction.csv`

### 4. Chạy Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Demo App

App gồm 3 tab chính:

- **🔮 Predict Transaction** — Nhập thông tin giao dịch thủ công, chạy full pipeline và cho ra fraud probability
- **📋 Batch Scoring** — Upload file CSV, score hàng loạt và download kết quả
- **📊 EDA** — Phân tích trực quan dataset: label distribution, transaction amount, fraud rate theo giờ/category

---

## 📈 Kết quả

Model được đánh giá trên **time-based split** (tránh data leakage):

| Metric | Score |
|---|---|
| AUC-ROC | ~0.93 |
| Threshold mặc định | 0.5 |

