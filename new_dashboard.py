import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime, timedelta

# --- CONFIGURATION & PATHS ---
PAGE_TITLE = "Sentinel Dashboard"
PAGE_ICON = "🛡️"
LAYOUT = "wide"
DB_PATH = Path("data/data_drift/drift_history.db")
STATE_PATH = Path("data/data_drift/system_state.json")

try:
    from services.rebase_workflow import RebaseWorkflow, RebaseMethod
    from services.system_state_tracker import SystemStateTracker
    from services.audit_log import SentinelAuditLog
    from services.model_registry import ModelRegistry
    from services.alert_utils import SentinelAlert
    REBASE_AVAILABLE = True
    AUDIT_AVAILABLE = True
    REGISTRY_AVAILABLE = True
    ALERTS_AVAILABLE = True
except ImportError:
    REBASE_AVAILABLE = False
    AUDIT_AVAILABLE = False
    REGISTRY_AVAILABLE = False
    ALERTS_AVAILABLE = False

# --- SETUP PAGE ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# --- GLOBAL STYLES ---
st.markdown("""
<style>
    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* ===== HEADER BAR ===== */
    .sentinel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        margin-bottom: 0.5rem;
    }
    .sentinel-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        letter-spacing: -0.02em;
    }

    /* ===== TAB NAVIGATION ===== */
    .nav-tabs {
        display: flex;
        gap: 0.25rem;
    }
    .nav-tabs a {
        color: #94a3b8;
        text-decoration: none;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    .nav-tabs a:hover {
        color: #e2e8f0;
        background: rgba(148, 163, 184, 0.1);
    }
    .nav-tabs a.active {
        color: #60a5fa;
        border-bottom: 2px solid #60a5fa;
        border-radius: 6px 6px 0 0;
    }

    /* ===== SECTION TITLE ===== */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 1rem 0 0.75rem 0;
    }

    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] > div {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }
    [data-testid="stMetricDelta"] > div {
        font-weight: 600;
    }

    /* ===== PANEL CARDS ===== */
    .panel-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        height: 100%;
    }
    .panel-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 1rem 0;
    }

    /* ===== EXECUTION HISTORY LIST ===== */
    .exec-entry {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.35rem;
        background: rgba(51, 65, 85, 0.4);
        transition: background 0.15s ease;
        cursor: pointer;
    }
    .exec-entry:hover {
        background: rgba(51, 65, 85, 0.7);
    }
    .exec-entry .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .exec-entry .dot.pass { background-color: #22c55e; }
    .exec-entry .dot.fail { background-color: #ef4444; }
    .exec-entry .ts {
        color: #94a3b8;
        font-size: 0.82rem;
        flex-grow: 1;
    }
    .exec-entry .chevron {
        color: #475569;
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }

    /* ===== REBASE BUTTON ===== */
    .rebase-btn-wrap {
        margin-top: 0.5rem;
    }
    .rebase-btn-wrap button {
        width: 100%;
        font-size: 0.75rem !important;
    }

    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== PLOTLY CONTAINER FIX ===== */
    [data-testid="stPlotlyChart"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* ===== PLACEHOLDER PAGES ===== */
    .placeholder-page {
        text-align: center;
        padding: 6rem 2rem;
    }
    .placeholder-page h2 {
        color: #f1f5f9;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .placeholder-page p {
        color: #64748b;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATA LAYER                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def get_db_connection():
    """Establishes connection to the SQLite database."""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_system_state():
    """Loads the current system state from the JSON file."""
    if not STATE_PATH.exists():
        return {"state": "NOMINAL", "last_reason": "No state file found"}
    try:
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
        query = "SELECT * FROM drift_logs ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        conn.close()
        return df
    except Exception as e:
        if conn:
            conn.close()
        return pd.DataFrame()


# ╔══════════════════════════════════════════════════════════════╗
# ║  NAVIGATION                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

TABS = ["Dashboard", "History", "Model Registry", "Settings"]


def render_header():
    """Renders the top header bar with title and navigation tabs."""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Dashboard"

    active = st.session_state.active_tab
    tab_links = ""
    for tab in TABS:
        cls = "active" if tab == active else ""
        tab_links += f'<a class="{cls}" href="?tab={tab}" target="_self">{tab}</a>'

    st.markdown(f"""
    <div class="sentinel-header">
        <h1>Sentinel Dashboard</h1>
        <div class="nav-tabs">
            {tab_links}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Process tab clicks via query params
    params = st.query_params
    tab_param = params.get("tab", None)
    if tab_param and tab_param in TABS:
        if st.session_state.active_tab != tab_param:
            st.session_state.active_tab = tab_param
            st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  DASHBOARD TAB — OVERVIEW                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def render_overview_metrics(df):
    """Renders the 5 overview metric cards matching the reference design."""
    if df.empty:
        st.warning("No data available to calculate metrics.")
        return

    latest = df.iloc[0]

    # Failure rate (last 10 runs)
    recent = df.head(10)
    fail_count = recent[recent['status'] == 'FAIL'].shape[0]
    fail_rate = (fail_count / len(recent)) * 100

    # System state
    system_state = load_system_state()
    state_val = system_state.get("state", "NOMINAL")

    # Next retraining
    retrain_val = "Unknown"
    retrain_label = "Next Retraining"
    try:
        import all_config as config
        interval_days = getattr(config, 'RETRAINING_INTERVAL_DAYS', 7)
        last_train = datetime.now()
        if AUDIT_AVAILABLE:
            audit = SentinelAuditLog()
            entries = audit.query(category="training", action="success", limit=1)
            if entries:
                last_train = datetime.fromisoformat(entries[0]['timestamp'])
        next_train = last_train + timedelta(days=interval_days)
        remaining = next_train - datetime.now()
        if remaining.total_seconds() < 0:
            retrain_val = "Overdue"
        else:
            retrain_val = f"{remaining.days}d {int(remaining.seconds / 3600)}h"
            retrain_label = f"Next Retraining"
    except ImportError:
        pass

    # --- Render the 5 cards ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        state_icons = {"NOMINAL": "🟢", "WARNING": "🟡", "RED": "🔴", "REBASE_IN_PROGRESS": "🔄"}
        icon = state_icons.get(state_val, "❓")
        st.metric(label="System State", value=f"{icon} {state_val}")
        # Rebase button nestled inside the System State card
        if REBASE_AVAILABLE:
            if st.button("🔄 Rebase System", key="rebase_btn", use_container_width=True):
                st.session_state['show_rebase_wizard'] = True
                st.rerun()

    with col2:
        status = latest['status']
        dot = "🟢" if status == "PASS" else "🔴"
        st.metric(label="Latest Status", value=f"{dot} {status}")

    with col3:
        score = latest['drift_score']
        st.metric(label="Drift Score", value=f"↑ {score:.2f}%")

    with col4:
        st.metric(label=f"Failure Rate ({len(recent)})", value=f"↑ {fail_rate:.0f}%")

    with col5:
        st.metric(label=retrain_label, value=f"⏳ {retrain_val}")


def render_drift_chart(df):
    """Renders the Drift Trends Over Time chart (left panel)."""
    if df.empty:
        return

    chart_df = df.sort_values('timestamp')

    fig = go.Figure()

    # Drift score line (blue)
    fig.add_trace(go.Scatter(
        x=chart_df['timestamp'],
        y=chart_df['drift_score'],
        mode='lines+markers',
        name='Drift Score',
        line=dict(color='#3b82f6', width=2.5),
        marker=dict(size=5, color='#3b82f6'),
    ))

    # Threshold line (red)
    threshold = chart_df['threshold'].iloc[0] if not chart_df.empty else 0.5
    fig.add_trace(go.Scatter(
        x=chart_df['timestamp'],
        y=[threshold] * len(chart_df),
        mode='lines',
        name='Threshold',
        line=dict(color='#ef4444', width=2, dash='dash'),
    ))

    # Confidence interval band
    if 'ci_low' in chart_df.columns and 'ci_high' in chart_df.columns:
        ci = chart_df.dropna(subset=['ci_low', 'ci_high'])
        if not ci.empty:
            fig.add_trace(go.Scatter(
                x=ci['timestamp'], y=ci['ci_high'],
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=ci['timestamp'], y=ci['ci_low'],
                mode='lines', line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.12)',
                name='90% CI',
            ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Y",
        hovermode="x unified",
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font=dict(color='#94a3b8'),
        xaxis=dict(
            gridcolor='#334155',
            linecolor='#334155',
            tickformat='%b %d',
        ),
        yaxis=dict(
            gridcolor='#334155',
            linecolor='#334155',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color='#94a3b8'),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_execution_history(df):
    """Renders the Execution History sidebar panel using an HTML component."""
    if df.empty:
        entries_html = '<p style="color:#64748b; font-size:0.9rem;">No data yet.</p>'
    else:
        recent = df.head(10)
        entries_html = ""
        for _, row in recent.iterrows():
            dot_color = "#22c55e" if row['status'] == "PASS" else "#ef4444"
            ts = row['timestamp'].strftime('%b %d, %Y, %I:%M:%S %p') if pd.notna(row['timestamp']) else "—"
            entries_html += f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        padding:0.6rem 0.7rem; border-radius:8px; margin-bottom:0.3rem;
                        background:rgba(51,65,85,0.4); cursor:pointer;"
                 onmouseover="this.style.background='rgba(51,65,85,0.7)'"
                 onmouseout="this.style.background='rgba(51,65,85,0.4)'">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%;
                             background:{dot_color}; margin-right:10px; flex-shrink:0;"></span>
                <span style="color:#94a3b8; font-size:0.82rem; flex-grow:1;">{ts}</span>
                <span style="color:#475569; font-size:0.9rem; margin-left:0.5rem;">›</span>
            </div>
            """

    full_html = f"""
    <div style="background-color:#1e293b; border:1px solid #334155; border-radius:10px;
                padding:1.25rem; font-family:sans-serif;">
        <h3 style="font-size:1rem; font-weight:600; color:#f1f5f9; margin:0 0 1rem 0;">Execution History</h3>
        {entries_html}
    </div>
    """
    # Calculate height based on number of entries
    num_entries = min(len(df), 10) if not df.empty else 1
    panel_height = 80 + num_entries * 48
    components.html(full_html, height=panel_height, scrolling=False)


def render_dashboard_tab():
    """Renders the complete Dashboard / Overview tab."""
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    df = load_data()

    # Check for rebase wizard
    if st.session_state.get('show_rebase_wizard', False) and REBASE_AVAILABLE:
        render_rebase_wizard()
        return

    render_overview_metrics(df)

    st.markdown("")  # spacer

    # Two-column layout: chart (left) + execution history (right)
    col_chart, col_history = st.columns([7, 3])

    with col_chart:
        st.markdown('<div class="section-title" style="font-size:1rem; margin-top:0;">Drift Trends Over Time</div>', unsafe_allow_html=True)
        render_drift_chart(df)

    with col_history:
        render_execution_history(df)


# ╔══════════════════════════════════════════════════════════════╗
# ║  REBASE WIZARD                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def render_rebase_wizard():
    """Renders the rebase wizard UI (carried over from dashboard.py)."""
    st.markdown('<div class="section-title">🔄 System Rebase Wizard</div>', unsafe_allow_html=True)
    st.markdown("Use this wizard to reset Sentinel after external changes to your model or data pipeline.")
    st.markdown("---")

    workflow = RebaseWorkflow()
    tracker = SystemStateTracker()

    # Check if rebase is already in progress
    if tracker.is_rebase_in_progress():
        rebase_status = tracker.get_rebase_status()
        st.warning(f"⏳ Rebase in progress: {rebase_status.get('reason', 'Unknown')} → {rebase_status.get('method', 'Unknown')}")
        st.caption(f"Started at: {rebase_status.get('started_at', 'Unknown')}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Complete Rebase", use_container_width=True):
                tracker.complete_rebase(success=True, details="Completed via dashboard")
                st.success("Rebase completed!")
                st.rerun()
        with c2:
            if st.button("❌ Cancel Rebase", use_container_width=True):
                tracker.cancel_rebase()
                st.info("Rebase cancelled.")
                st.rerun()
        return

    # Step 1: What changed?
    st.markdown("### Step 1: What changed?")
    change_types = workflow.get_change_types()
    change_type = st.radio(
        "Select what changed in your system:",
        options=[ct["value"] for ct in change_types],
        format_func=lambda x: next((ct["label"] for ct in change_types if ct["value"] == x), x),
        horizontal=False,
        key="rebase_change_type"
    )
    selected_change = next((ct for ct in change_types if ct["value"] == change_type), None)
    if selected_change:
        st.caption(f"ℹ️ {selected_change['description']}")

    st.markdown("---")

    # Step 2: How to proceed?
    st.markdown("### Step 2: How should Sentinel adapt?")
    methods = workflow.get_options_for_change(change_type)
    if not methods:
        st.error("No valid rebase methods for this change type.")
        return

    method = st.radio(
        "Select rebase method:",
        options=[m["value"] for m in methods],
        format_func=lambda x: next((m["label"] for m in methods if m["value"] == x), x),
        horizontal=False,
        key="rebase_method"
    )
    selected_method = next((m for m in methods if m["value"] == method), None)
    if selected_method:
        st.caption(f"ℹ️ {selected_method['description']}")

    # Configuration for specific methods
    config_data = {}
    if method == "new_training_data":
        st.markdown("---")
        st.markdown("### Configuration")
        training_path = st.text_input(
            "Training data path:",
            placeholder="/path/to/training/data",
            help="Absolute path to the directory containing training data"
        )
        if training_path:
            config_data["training_data_path"] = training_path

    st.markdown("---")

    # Step 3: Execute
    st.markdown("### Step 3: Execute Rebase")
    can_proceed = True
    if method == "new_training_data" and not config_data.get("training_data_path"):
        st.warning("⚠️ Please provide the training data path.")
        can_proceed = False

    if st.button("🚀 Start Rebase", use_container_width=True, type="primary", disabled=not can_proceed):
        success = workflow.start(change_type=change_type, method=method, config_data=config_data)
        if success:
            if method == "keep_baseline":
                st.success("✅ Rebase complete! System returned to NOMINAL.")
            else:
                st.info("📊 Rebase started. Check the progress below.")
            st.rerun()
        else:
            st.error("❌ Failed to start rebase. Check logs for details.")

    if st.button("← Back to Dashboard", use_container_width=True):
        if 'show_rebase_wizard' in st.session_state:
            del st.session_state['show_rebase_wizard']
        st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  STUB TABS                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def render_history_tab():
    """Placeholder for the History tab."""
    st.markdown("""
    <div class="placeholder-page">
        <h2>📜 History</h2>
        <p>Execution history and audit logs will be displayed here.</p>
    </div>
    """, unsafe_allow_html=True)


def render_model_registry_tab():
    """Placeholder for the Model Registry tab."""
    st.markdown("""
    <div class="placeholder-page">
        <h2>📦 Model Registry</h2>
        <p>Model version management and deployment history will be displayed here.</p>
    </div>
    """, unsafe_allow_html=True)


def render_settings_tab():
    """Placeholder for the Settings tab."""
    st.markdown("""
    <div class="placeholder-page">
        <h2>⚙️ Settings</h2>
        <p>Configuration and preferences will be displayed here.</p>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SETUP REQUIRED PAGE                                          ║
# ╚══════════════════════════════════════════════════════════════╝

def render_setup_required():
    """Renders a 'Setup Required' page for first-time users."""
    st.title("🛡️ Welcome to Sentinel")
    st.markdown("---")
    st.markdown("""
    ## 🆕 Initial Setup Required

    Sentinel needs to be configured before it can start monitoring.
    The setup wizard will:

    - 📂 **Load your training data** and validate it
    - 🧠 **Register your production model**
    - ⚖️ **Calibrate drift thresholds** from your data
    - 🎯 **Create or link a Golden Set** for benchmarking
    - 🧪 **Distill your model** for efficient feature extraction
    """)
    st.markdown("---")
    st.markdown("### How to Run Setup")
    st.code("docker compose exec sentinel python setup.py", language="bash")
    st.markdown("""
    > **Tip:** If the container isn't running yet, use `docker compose run --rm sentinel python setup.py` instead.

    ### Accessing External Data
    Files from your host machine are mounted at **`/host-data/`** inside the container.
    """)
    st.code("HOST_DATA_DIR=/home/your-username  # or /home/ec2-user on AWS", language="bash")
    st.markdown("""
    Once setup completes, **restart the container** and this page will be replaced by the monitoring dashboard:
    """)
    st.code("docker compose down && docker compose up", language="bash")
    st.caption("Sentinel will check for the initialization marker at `data/.sentinel_initialized`")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # First-run detection
    marker_file = Path("data/.sentinel_initialized")
    if not marker_file.exists():
        render_setup_required()
        return

    render_header()

    active = st.session_state.get("active_tab", "Dashboard")

    if active == "Dashboard":
        render_dashboard_tab()
    elif active == "History":
        render_history_tab()
    elif active == "Model Registry":
        render_model_registry_tab()
    elif active == "Settings":
        render_settings_tab()


if __name__ == "__main__":
    main()
