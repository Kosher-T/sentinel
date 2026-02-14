#!/bin/bash
# Sentinel Container Entrypoint
# Handles first-run detection and service orchestration

# Ensure venv is on PATH and PYTHONPATH
export VIRTUAL_ENV="/app/venv"
export PATH="/app/venv/bin:$PATH"
export PYTHONPATH="/app:/app/venv/lib/python3.12/site-packages"

MARKER_FILE="/app/data/.sentinel_initialized"

if [ -f "$MARKER_FILE" ]; then
    echo "✅ Sentinel initialized. Starting services..."

    # Start sentinel_watch in the background
    python sentinel_watch.py &
    WATCH_PID=$!
    echo "🔭 sentinel_watch started (PID: $WATCH_PID)"

    # Start the dashboard in the foreground
    echo "📊 Starting dashboard on port 8501..."
    exec python -m streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
else
    echo "🆕 First run detected. Starting dashboard in setup mode..."
    echo "   Run setup with: docker compose exec sentinel python setup.py"

    # Start dashboard (it will show "Setup Required" page)
    exec python -m streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
fi
