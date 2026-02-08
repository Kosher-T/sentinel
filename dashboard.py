import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime

# --- CONFIGURATION & PATHS ---
PAGE_TITLE = "Sentinel Monitoring"
PAGE_ICON = "🛡️"
LAYOUT = "wide"
DB_PATH = Path("data/data_drift/drift_history.db")
STATE_PATH = Path("data/data_drift/system_state.json")

# --- SETUP PAGE ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# --- STYLING ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Metric Card Styling - Darker boxes */
    [data-testid="stMetric"] {
        background-color: #1e293b; /* Dark slate background */
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Force white text for metrics */
    [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] > div {
        color: #94a3b8 !important; /* Lighter grey for label */
        font-size: 0.9rem;
    }

    /* Adjust delta colors for dark background visibility */
    [data-testid="stMetricDelta"] > div {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LAYER ---

def get_db_connection():
    """Establishes connection to the SQLite database."""
    if not DB_PATH.exists():
        st.error(f"⚠️ Database not found at: {DB_PATH}")
        return None
    # Use check_same_thread=False for streamlit compatibility
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def load_system_state():
    """Loads the current system state from the JSON file."""
    if not STATE_PATH.exists():
        return {"state": "NOMINAL", "last_reason": "No state file found"}
    
    try:
        import json
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {"state": "NOMINAL", "last_reason": "Error reading state"}

def load_data():
    """Loads drift logs from the database into a Pandas DataFrame."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        # We don't use st.cache_data here to ensure every 'rerun' hits the DB
        query = "SELECT * FROM drift_logs ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

# --- COMPONENT: METRICS ---

def render_metrics(df):
    """Renders the top-level KPI metrics including system state."""
    if df.empty:
        st.warning("No data available to calculate metrics.")
        return

    # Ensure we are looking at the absolute latest entry
    latest_run = df.iloc[0]
    
    # Calculate Failure Rate (Last 10 Runs)
    recent_df = df.head(10)
    fail_count = recent_df[recent_df['status'] == 'FAIL'].shape[0]
    fail_rate = (fail_count / 10) * 100
    
    # Load system state
    system_state = load_system_state()
    state_value = system_state.get("state", "NOMINAL")
    state_reason = system_state.get("last_reason", "")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # System State with color coding
        state_icons = {"NOMINAL": "🟢", "WARNING": "🟡", "RED": "🔴"}
        state_icon = state_icons.get(state_value, "❓")
        st.metric(label="System State", value=f"{state_icon} {state_value}", delta=state_reason[:30] + "..." if len(state_reason) > 30 else state_reason, delta_color="off")
    
    with col2:
        status_color = "normal" if latest_run['status'] == "PASS" else "inverse"
        st.metric(label="Latest Status", value=latest_run['status'], delta="System Check", delta_color=status_color)
    
    with col3:
        st.metric(label="Drift Score", value=f"{latest_run['drift_score']:.2f}%", delta=f"Limit: {latest_run['threshold']}%", delta_color="inverse")

    with col4:
        st.metric(label="Failure Rate (L10)", value=f"{fail_rate:.0f}%", delta="Historical Trend", delta_color="off")

    with col5:
        # Displaying local time to show it's active
        check_time = latest_run['timestamp'].strftime('%H:%M:%S')
        check_date = latest_run['timestamp'].strftime('%Y-%m-%d')
        st.metric(label=f"Last Sync ({check_date})", value=check_time)

# --- COMPONENT: CHARTS ---

def render_drift_chart(df):
    """Renders the main time-series drift chart using Plotly."""
    if df.empty:
        return

    st.subheader("📉 Drift Trends Over Time")
    
    fig = go.Figure()

    # Drift Score Line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['drift_score'],
        mode='lines+markers',
        name='Drift Score',
        line=dict(color='#4F46E5', width=3),
        marker=dict(size=6)
    ))

    # Threshold Line
    threshold_val = df['threshold'].iloc[0] if not df.empty else 0.5
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[threshold_val] * len(df),
        mode='lines',
        name='Threshold',
        line=dict(color='#EF4444', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Drift Score",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')

def render_history_table(df):
    """Renders the detailed history table with visual enhancements."""
    st.subheader("📜 Execution History")
    
    if df.empty:
        st.info("No history found.")
        return

    display_df = df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        display_df,
        width='stretch',
        column_config={
            "drift_score": st.column_config.ProgressColumn(
                "Drift Magnitude",
                help="Visual representation of drift score",
                format="%.2f",
                min_value=0,
                max_value=1, 
            ),
            "status": st.column_config.TextColumn(
                "Status",
                help="Pass/Fail status",
                validate="^(PASS|FAIL)$"
            ),
        },
        hide_index=True
    )

# --- COMPONENT: SIDEBAR ---

def render_sidebar():
    """Renders the sidebar configuration."""
    st.sidebar.title(f"{PAGE_ICON} Sentinel")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Settings")
    limit = st.sidebar.slider("Records to fetch", min_value=10, max_value=500, value=100)
    
    auto_refresh = st.sidebar.checkbox("Live Monitoring (5s)", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Project Info")
    st.sidebar.code("""
    root/
    └── data/
        └── drift_history.db
    """, language="text")
    
    return limit, auto_refresh

# --- MAIN APP LOGIC ---

def main():
    limit, auto_refresh = render_sidebar()

    # Force a reload from disk every time main is called
    df = load_data()
    
    if not df.empty:
        df = df.head(limit)

    st.title("Sentinel Data Drift Dashboard")
    st.markdown(f"Monitoring **{len(df)}** recent checkpoints from `{DB_PATH}`")
    st.markdown("---")

    render_metrics(df)
    render_drift_chart(df)
    render_history_table(df)

    # 7. Auto Refresh Logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()