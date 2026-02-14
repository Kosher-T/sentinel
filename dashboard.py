import streamlit as st
import sqlite3
import json
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

try:
    from services.rebase_workflow import RebaseWorkflow, RebaseMethod
    from services.system_state_tracker import SystemStateTracker
    from services.audit_log import SentinelAuditLog
    REBASE_AVAILABLE = True
    AUDIT_AVAILABLE = True
except ImportError:
    REBASE_AVAILABLE = False
    AUDIT_AVAILABLE = False

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
        state_icons = {"NOMINAL": "🟢", "WARNING": "🟡", "RED": "🔴", "REBASE_IN_PROGRESS": "🔄"}
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

def render_root_cause():
    """Renders drift root cause breakdown from the drift_logs DB or audit log."""
    # Try to load root cause from the drift_logs DB first
    root_cause = None
    source_timestamp = None
    
    conn = get_db_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("SELECT root_cause_json, timestamp FROM drift_logs WHERE status='FAIL' AND root_cause_json IS NOT NULL ORDER BY timestamp DESC LIMIT 1")
            row = c.fetchone()
            if row and row[0]:
                root_cause = json.loads(row[0])
                source_timestamp = row[1]
            conn.close()
        except Exception:
            if conn:
                conn.close()
    
    # Fallback to audit log if DB doesn't have root cause data
    if not root_cause and AUDIT_AVAILABLE:
        audit = SentinelAuditLog()
        fail_entries = audit.query(category="drift", action="check_fail", limit=1)
        if fail_entries:
            entry = fail_entries[0]
            details = entry.get("details", {})
            if details and "primary_drivers" in details:
                root_cause = details
                source_timestamp = entry.get("timestamp", "N/A")
    
    if not root_cause or "primary_drivers" not in root_cause:
        return
    
    st.subheader("🔍 Drift Root Cause Analysis")
    
    # Pattern indicator
    pattern = root_cause.get("drift_pattern", "unknown")
    drifting = root_cause.get("drifting_components", 0)
    total = root_cause.get("total_components", 0)
    pattern_config = {
        "localized": {"icon": "🎯", "color": "#F59E0B", "desc": "Drift concentrated in few features — likely a specific data subset changed"},
        "moderate": {"icon": "🔶", "color": "#F97316", "desc": "Drift across several features — possibly a gradual distribution shift"},
        "widespread": {"icon": "🌊", "color": "#EF4444", "desc": "Most features drifting — significant change in data characteristics"},
    }
    p = pattern_config.get(pattern, {"icon": "❓", "color": "#6B7280", "desc": "Unknown"})
    
    col_pattern, col_info = st.columns([1, 2])
    with col_pattern:
        st.metric(label="Drift Pattern", value=f"{p['icon']} {pattern.upper()}", delta=f"{drifting}/{total} components drifting", delta_color="off")
    with col_info:
        st.caption(p["desc"])
        if source_timestamp:
            st.caption(f"From: {source_timestamp}")
    
    # Bar chart of primary drivers
    drivers = root_cause.get("primary_drivers", [])
    if drivers:
        components = [f"Component {d['component']}" for d in drivers]
        scores = [d["drift_score"] for d in drivers]
        
        colors = ["#EF4444" if s > 0.5 else "#F59E0B" if s > 0.2 else "#22C55E" for s in scores]
        
        fig = go.Figure(go.Bar(
            x=scores,
            y=components,
            orientation='h',
            marker_color=colors,
            text=[f"{s:.4f}" for s in scores],
            textposition='auto',
        ))
        fig.update_layout(
            title="Top Drift Drivers (by contribution)",
            xaxis_title="Drift Score",
            yaxis_title="",
            margin=dict(l=0, r=0, t=40, b=0),
            height=200,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed component metrics table
    per_component = root_cause.get("per_component", [])
    if per_component:
        st.markdown("**Component Diagnostic Detail**")
        
        table_data = []
        for c in per_component:
            ks_p = c.get('ks_pvalue', None)
            sig = "✅ Yes" if (ks_p is not None and ks_p < 0.05) else "—"
            
            table_data.append({
                "Component": f"PC-{c['component']}",
                "Drift Score": round(c.get('drift_score', 0), 4),
                "Variance %": f"{c.get('explained_variance', 0):.1%}",
                "Wasserstein": round(c.get('wasserstein', 0), 4),
                "KL Div": round(c.get('kl_divergence', 0), 4),
                "Mean Shift": round(c.get('mean_shift', 0), 4),
                "Var Ratio": round(c.get('variance_ratio', 1), 4),
                "Skew Δ": round(c.get('skewness_delta', 0), 4),
                "KS Stat": round(c.get('ks_statistic', 0), 4),
                "KS Sig?": sig,
            })
        
        df_detail = pd.DataFrame(table_data)
        st.dataframe(df_detail, use_container_width=True, hide_index=True)


def render_root_cause_trends():
    """Renders historical root cause trends — which components repeatedly drive drift."""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        c = conn.cursor()
        c.execute("SELECT timestamp, root_cause_json FROM drift_logs WHERE status='FAIL' AND root_cause_json IS NOT NULL ORDER BY timestamp DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
    except Exception:
        if conn:
            conn.close()
        return
    
    if len(rows) < 2:
        return  # Need at least 2 FAIL entries to show trends
    
    st.subheader("📊 Root Cause Trends")
    st.caption("Which PCA components are repeatedly driving drift across recent failures")
    
    # Build heatmap data: timestamps × components
    heatmap_data = []  # {timestamp, component, drift_score}
    driver_counts = {}  # component -> count of times in top 3
    
    for timestamp, rc_json in rows:
        try:
            rc = json.loads(rc_json)
        except (json.JSONDecodeError, TypeError):
            continue
        
        per_component = rc.get("per_component", [])
        primary_drivers = rc.get("primary_drivers", [])
        
        # Track top driver appearances
        for d in primary_drivers:
            comp = d.get("component", 0)
            driver_counts[comp] = driver_counts.get(comp, 0) + 1
        
        # Build heatmap row
        for comp in per_component:
            heatmap_data.append({
                "Check": timestamp[:16],  # Trim to minutes
                "Component": f"PC-{comp['component']}",
                "Drift Score": comp.get("drift_score", 0),
            })
    
    if not heatmap_data:
        return
    
    # --- Heatmap ---
    df_heat = pd.DataFrame(heatmap_data)
    
    fig = go.Figure(data=go.Heatmap(
        x=df_heat["Check"].unique(),
        y=sorted(df_heat["Component"].unique()),
        z=df_heat.pivot_table(index="Component", columns="Check", values="Drift Score", aggfunc="first").reindex(
            index=sorted(df_heat["Component"].unique())
        ).values,
        colorscale=[
            [0, "#1e293b"],     # Dark slate — no drift
            [0.2, "#22C55E"],   # Green — low
            [0.5, "#F59E0B"],   # Amber — moderate
            [1.0, "#EF4444"],   # Red — high
        ],
        text=df_heat.pivot_table(index="Component", columns="Check", values="Drift Score", aggfunc="first").reindex(
            index=sorted(df_heat["Component"].unique())
        ).round(3).values,
        texttemplate="%{text}",
        hovertemplate="%{y} at %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Component Drift Scores Across Recent Failures",
        xaxis_title="Drift Check",
        yaxis_title="",
        margin=dict(l=0, r=0, t=40, b=0),
        height=max(200, len(df_heat["Component"].unique()) * 35 + 80),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Repeat Offenders ---
    if driver_counts:
        st.markdown("**🔄 Repeat Offenders** — Components most frequently in the top 3 drivers")
        sorted_drivers = sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)
        total_checks = len(rows)
        
        offender_data = []
        for comp, count in sorted_drivers:
            pct = (count / total_checks) * 100
            offender_data.append({
                "Component": f"PC-{comp}",
                "Appearances": f"{count}/{total_checks}",
                "Frequency": f"{pct:.0f}%",
                "Severity": "🔴 Persistent" if pct >= 75 else "🟡 Recurring" if pct >= 40 else "🟢 Occasional",
            })
        
        df_offenders = pd.DataFrame(offender_data)
        st.dataframe(df_offenders, use_container_width=True, hide_index=True)

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
    
    # Rebase System Section
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ System Rebase")
    rebase_clicked = st.sidebar.button("🔄 Rebase System", use_container_width=True, disabled=not REBASE_AVAILABLE)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Project Info")
    st.sidebar.code("""
    root/
    └── data/
        └── drift_history.db
    """, language="text")
    
    return limit, auto_refresh, rebase_clicked


# --- COMPONENT: REBASE WIZARD ---

def render_rebase_wizard():
    """Renders the rebase wizard UI."""
    st.subheader("🔄 System Rebase Wizard")
    st.markdown("Use this wizard to reset Sentinel after external changes to your model or data pipeline.")
    st.markdown("---")
    
    workflow = RebaseWorkflow()
    tracker = SystemStateTracker()
    
    # Check if rebase is already in progress
    if tracker.is_rebase_in_progress():
        rebase_status = tracker.get_rebase_status()
        st.warning(f"⏳ Rebase in progress: {rebase_status.get('reason', 'Unknown')} → {rebase_status.get('method', 'Unknown')}")
        st.caption(f"Started at: {rebase_status.get('started_at', 'Unknown')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Complete Rebase", use_container_width=True):
                tracker.complete_rebase(success=True, details="Completed via dashboard")
                st.success("Rebase completed!")
                st.rerun()
        with col2:
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
    
    # Show description for selected change
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
    
    # Show description for selected method
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
    
    # Validation
    can_proceed = True
    if method == "new_training_data" and not config_data.get("training_data_path"):
        st.warning("⚠️ Please provide the training data path.")
        can_proceed = False
    
    if st.button("🚀 Start Rebase", use_container_width=True, type="primary", disabled=not can_proceed):
        success = workflow.start(
            change_type=change_type,
            method=method,
            config_data=config_data
        )
        
        if success:
            if method == "keep_baseline":
                st.success("✅ Rebase complete! System returned to NOMINAL.")
            else:
                st.info("📊 Rebase started. Check the progress below.")
            st.rerun()
        else:
            st.error("❌ Failed to start rebase. Check logs for details.")
    
    # Cancel button
    if st.button("← Back to Dashboard", use_container_width=True):
        if 'show_rebase_wizard' in st.session_state:
            del st.session_state['show_rebase_wizard']
        st.rerun()

# --- COMPONENT: AUDIT LOG ---

# Category colors for visual distinction
CATEGORY_COLORS = {
    "drift": "#4F46E5",       # Indigo
    "data": "#0891B2",        # Cyan
    "training": "#D97706",    # Amber
    "decay": "#7C3AED",       # Purple
    "deployment": "#059669",  # Emerald
    "baseline": "#6366F1",    # Indigo-light
    "alert": "#DC2626",       # Red
    "state": "#6B7280",       # Gray
    "rebase": "#2563EB",      # Blue
}

CATEGORY_ICONS = {
    "drift": "📊",
    "data": "📁",
    "training": "🏋️",
    "decay": "🔬",
    "deployment": "🚀",
    "baseline": "📐",
    "alert": "🚨",
    "state": "🔄",
    "rebase": "♻️",
}

def render_audit_log():
    """Renders the audit log timeline with filters."""
    if not AUDIT_AVAILABLE:
        return
    
    st.markdown("---")
    st.subheader("📝 Audit Log")
    
    audit = SentinelAuditLog()
    
    # Filters row
    col_filter, col_limit = st.columns([3, 1])
    
    with col_filter:
        all_categories = list(CATEGORY_COLORS.keys())
        selected_categories = st.multiselect(
            "Filter by category",
            options=all_categories,
            default=[],
            format_func=lambda x: f"{CATEGORY_ICONS.get(x, '')} {x.title()}",
            key="audit_category_filter"
        )
    
    with col_limit:
        audit_limit = st.selectbox("Entries", options=[25, 50, 100, 200], index=1, key="audit_limit")
    
    # Fetch entries
    if selected_categories:
        # Query each selected category and merge
        entries = []
        for cat in selected_categories:
            entries.extend(audit.query(category=cat, limit=audit_limit))
        # Sort by timestamp descending
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        entries = entries[:audit_limit]
    else:
        entries = audit.get_timeline(limit=audit_limit)
    
    if not entries:
        st.info("No audit entries recorded yet. Entries will appear after Sentinel runs.")
        return
    
    # Summary counts
    counts = audit.get_category_counts()
    if counts:
        count_cols = st.columns(min(len(counts), 6))
        for i, (cat, count) in enumerate(list(counts.items())[:6]):
            icon = CATEGORY_ICONS.get(cat, "📌")
            with count_cols[i % len(count_cols)]:
                st.metric(label=f"{icon} {cat.title()}", value=count)
    
    st.markdown("")
    
    # Build dataframe for display
    rows = []
    for entry in entries:
        details_str = ""
        if entry["details"]:
            details_str = ", ".join(f"{k}={v}" for k, v in entry["details"].items())
        
        rows.append({
            "Time": entry["timestamp"],
            "Category": f"{CATEGORY_ICONS.get(entry['category'], '📌')} {entry['category']}",
            "Action": entry["action"],
            "Status": "✅" if entry["status"] == "success" else "❌" if entry["status"] == "failure" else "⚠️",
            "Details": details_str,
        })
    
    df_audit = pd.DataFrame(rows)
    
    st.dataframe(
        df_audit,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time": st.column_config.TextColumn("Time", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Details": st.column_config.TextColumn("Details", width="large"),
        }
    )


# --- MAIN APP LOGIC ---

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
    
    st.markdown("""
    ### How to Run Setup
    
    Open a terminal and run:
    """)
    
    st.code("docker compose exec sentinel python setup.py", language="bash")
    
    st.markdown("""
    > **Tip:** If the container isn't running yet, use `docker compose run --rm sentinel python setup.py` instead.
    
    ### Accessing External Data
    
    Files from your host machine are mounted at **`/host-data/`** inside the container.
    Configure which directory to mount by editing `.env`:
    """)
    
    st.code("HOST_DATA_DIR=/home/your-username  # or /home/ec2-user on AWS", language="bash")
    
    st.markdown("""
    For example, if your training data is at `/home/user/datasets/training/`, 
    you'd enter `/host-data/datasets/training/` when prompted during setup.
    
    Once setup completes, **restart the container** and this page will be replaced by the monitoring dashboard:
    """)

    
    st.code("docker compose down && docker compose up", language="bash")
    
    st.markdown("---")
    st.caption("Sentinel will check for the initialization marker at `data/.sentinel_initialized`")


def main():
    # First-run detection
    marker_file = Path("data/.sentinel_initialized")
    if not marker_file.exists():
        render_setup_required()
        return
    
    limit, auto_refresh, rebase_clicked = render_sidebar()
    
    # Handle rebase button click
    if rebase_clicked:
        st.session_state['show_rebase_wizard'] = True
    
    # Check if we should show rebase wizard
    if st.session_state.get('show_rebase_wizard', False) and REBASE_AVAILABLE:
        st.title("Sentinel System Rebase")
        render_rebase_wizard()
        return  # Don't show normal dashboard during rebase

    # Force a reload from disk every time main is called
    df = load_data()
    
    if not df.empty:
        df = df.head(limit)

    st.title("Sentinel Data Drift Dashboard")
    st.markdown(f"Monitoring **{len(df)}** recent checkpoints from `{DB_PATH}`")
    st.markdown("---")

    render_metrics(df)
    render_drift_chart(df)
    render_root_cause()
    render_root_cause_trends()
    render_history_table(df)
    render_audit_log()

    # Auto Refresh Logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()