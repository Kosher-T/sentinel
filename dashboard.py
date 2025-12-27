import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
from pathlib import Path

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(
    page_title="Sentinel Monitor",
    page_icon="🛰️",
    layout="wide"
)

# Setup paths to find all_config.py in the root
# Using resolve().parent ensures we are looking at the actual project root
file_path = Path(__file__).resolve()
project_root = file_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import all_config as config
    DB_PATH = config.DRIFT_HISTORY_DB
except ImportError:
    # Fallback if config is not found during initial boot
    DB_PATH = Path("data/data_drift/drift_history.db")

def load_data():
    """Reads the sqlite database into a Pandas DataFrame."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        # Use URI mode for read-only to prevent locking issues with the watcher
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM drift_logs", conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        # Don't let a DB error crash the whole UI
        return pd.DataFrame()

# Header
st.title("🛰️ Sentinel: VFI Model Monitor")
st.markdown("### Real-time Drift Detection & Self-Healing Log")

# Sidebar Status Area
with st.sidebar:
    st.header("System Status")
    if os.path.exists(DB_PATH):
        st.success("✅ Database Connected")
    else:
        st.error("❌ Database Not Found")
        st.caption(f"Searching: {DB_PATH}")
    
    if st.button('🔄 Refresh Dashboard'):
        st.rerun()

# Main Logic
df = load_data()

if df.empty:
    st.warning("Waiting for data...")
    st.info(f"The dashboard is active, but the log file is empty or hasn't been created yet.")
    st.caption(f"Target Path: {DB_PATH}")
else:
    # 1. Top Level Metrics
    latest_run = df.iloc[-1]
    last_score = latest_run['drift_score']
    last_status = latest_run['status']
    threshold = latest_run['threshold']

    col1, col2, col3 = st.columns(3)
    
    with col1:
        margin = threshold - last_score
        st.metric(label="Latest Drift Score", value=f"{last_score:.2f}%", delta=f"{margin:.2f}% Margin")
    
    with col2:
        st.markdown("**Drift Status**")
        if last_status == "FAIL":
            st.error(f"🚨 {last_status}")
        else:
            st.success(f"✅ {last_status}")
            
    with col3:
        st.metric(label="Drift Threshold", value=f"{threshold}%")

    # 2. The Chart
    st.markdown("### 📉 Drift Trend Over Time")
    chart_data = df[['timestamp', 'drift_score']].set_index('timestamp')
    st.line_chart(chart_data)

    # 3. Raw Log
    with st.expander("View Raw Logs"):
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)

    st.divider()
    st.caption(f"Connected to Sentinel Watcher | Database: {DB_PATH}")