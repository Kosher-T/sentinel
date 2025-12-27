import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
from pathlib import Path

# Setup paths to find all_config.py in the root
file_path = Path(__file__).resolve()
project_root = file_path.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

# Page Config
st.set_page_config(
    page_title="Sentinel Monitor",
    page_icon="🛰️",
    layout="wide"
)

# Header
st.title("🛰️ Sentinel: VFI Model Monitor")
st.markdown("### Real-time Drift Detection & Self-Healing Log")

# Path to the database from central config
DB_PATH = config.DRIFT_HISTORY_DB

def load_data():
    """Reads the sqlite database into a Pandas DataFrame."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame() # Return empty if no data yet
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM drift_logs", conn)
        conn.close()
        
        # Convert string timestamp to datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame()

# Auto-refresh logic
if st.button('🔄 Refresh Data'):
    st.rerun()

df = load_data()

if df.empty:
    st.warning(f"Waiting for data... Database not found or empty at: {DB_PATH}")
    st.info("Ensure the Sentinel Watcher has completed at least one run to initialize the database.")
else:
    # 1. Top Level Metrics
    latest_run = df.iloc[-1]
    last_score = latest_run['drift_score']
    last_status = latest_run['status']
    threshold = latest_run['threshold']

    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate margin (Higher score = worse, so positive delta if we are below threshold)
        margin = threshold - last_score
        st.metric(label="Latest Drift Score", value=f"{last_score:.2f}%", delta=f"{margin:.2f}% Margin")
    
    with col2:
        st.markdown("**System Status**")
        if last_status == "FAIL":
            st.error(f"🚨 {last_status}")
        else:
            st.success(f"✅ {last_status}")
            
    with col3:
        st.metric(label="Drift Threshold", value=f"{threshold}%")

    # 2. The Chart
    st.markdown("### 📉 Drift Trend Over Time")
    
    # Simple line chart using Streamlit's native component
    chart_data = df[['timestamp', 'drift_score']].set_index('timestamp')
    st.line_chart(chart_data)

    # 3. The Raw Log (Dataframe)
    with st.expander("View Raw Logs"):
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)

    # 4. System Info (Footer)
    st.divider()
    st.caption(f"Connected to Sentinel Watcher | Database: {DB_PATH} | Backbone: {config.EMBEDDING_MODEL_TYPE}")