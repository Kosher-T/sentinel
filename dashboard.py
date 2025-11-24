import streamlit as st
import pandas as pd
import sqlite3
import os
import time

# Page Config
st.set_page_config(
    page_title="Sentinel Monitor",
    page_icon="🛰️",
    layout="wide"
)

# Header
st.title("🛰️ Sentinel: VFI Model Monitor")
st.markdown("### Real-time Drift Detection & Self-Healing Log")

# Path to the database (Shared Volume)
DB_PATH = "temp_status/drift_history.db"

def load_data():
    """Reads the sqlite database into a Pandas DataFrame."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame() # Return empty if no data yet
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM drift_logs", conn)
    conn.close()
    
    # Convert string timestamp to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Auto-refresh logic (Simulates real-time monitoring)
if st.button('🔄 Refresh Data'):
    st.rerun()

df = load_data()

if df.empty:
    st.warning("Waiting for data... Run the Docker Monitor to generate logs.")
else:
    # 1. Top Level Metrics
    latest_run = df.iloc[-1]
    last_score = latest_run['drift_score']
    last_status = latest_run['status']
    threshold = latest_run['threshold']

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Latest Drift Score", value=f"{last_score:.2f}%", delta=f"{threshold - last_score:.2f}% Margin")
    
    with col2:
        if last_status == "FAIL":
            st.error(f"Status: {last_status}")
        else:
            st.success(f"Status: {last_status}")
            
    with col3:
        st.metric(label="Drift Threshold", value=f"{threshold}%")

    # 2. The Chart
    st.markdown("### 📉 Drift Trend Over Time")
    
    # We create a line chart of Score vs Time
    st.line_chart(df, x="timestamp", y="drift_score")

    # 3. The Raw Log (Dataframe)
    with st.expander("View Raw Logs"):
        st.dataframe(df.sort_values(by="timestamp", ascending=False))