import sqlite3
import pandas as pd
from datetime import datetime
import os

# Đường dẫn tới tệp cơ sở dữ liệu trong cùng thư mục dự án
DB_PATH = "sentinel.db"

def init_db():
    """Khởi tạo bảng cases nếu chưa tồn tại"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Tạo bảng với cột amount
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            probability REAL,
            prediction_label TEXT,
            card1 TEXT,
            addr1 TEXT,
            email TEXT,
            status TEXT DEFAULT 'Pending',
            investigator TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    ''')
    
    # Logic migration: Kiểm tra xem cột amount đã có chưa (cho các DB cũ)
    cursor.execute("PRAGMA table_info(cases)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'amount' not in columns:
        cursor.execute("ALTER TABLE cases ADD COLUMN amount REAL")
        
    conn.commit()
    conn.close()

def is_transaction_logged(transaction_id):
    """Kiểm tra xem transaction_id đã tồn tại trong DB chưa"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM cases WHERE transaction_id = ?", (str(transaction_id),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def save_case(transaction_id, amount, probability, prediction_label, card1="", addr1="", email="", investigator="System", status="Pending"):
    """Lưu một ca điều tra mới với dữ liệu quan hệ, kiểm tra trùng lặp"""
    if is_transaction_logged(transaction_id):
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO cases (transaction_id, amount, probability, prediction_label, card1, addr1, email, investigator, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (str(transaction_id), float(amount), float(probability), prediction_label, str(card1), str(addr1), str(email), investigator, status))
    conn.commit()
    conn.close()
    return True

def get_all_cases():
    """Lấy toàn bộ danh sách hồ sơ điều tra"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM cases ORDER BY created_at DESC", conn)
    except Exception:
        # Trường hợp bảng chưa có dữ liệu hoặc lỗi schema
        df = pd.DataFrame()
    conn.close()
    return df

def update_case_status(case_id, new_status, notes=""):
    """Cập nhật trạng thái và ghi chú cho hồ sơ"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE cases 
        SET status = ?, notes = ?
        WHERE id = ?
    ''', (new_status, notes, case_id))
    conn.commit()
    conn.close()

def get_case_by_id(case_id):
    """Lấy thông tin chi tiết một ca theo ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    return None

def get_related_cases(case_id):
    """Tìm các hồ sơ có cùng card1, addr1 hoặc email với case_id"""
    case = get_case_by_id(case_id)
    if not case: return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT * FROM cases 
        WHERE (card1 = ? OR addr1 = ? OR email = ?)
        AND id != ?
    """
    df = pd.read_sql_query(query, conn, params=(case['card1'], case['addr1'], case['email'], case_id))
    conn.close()
    return df

# Tự động khởi tạo ngay khi tệp được nạp
init_db()
