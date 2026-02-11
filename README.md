# 🛡️ Sentinel: Automated Model Drift Detection & Retraining System

**Sentinel** is an advanced machine learning monitoring system designed to detect data drift, automate model retraining, and ensure model performance in production environments without manual intervention.

It acts as a "self-healing" loop for your ML pipeline:
1.  **Monitors** incoming production data for distribution shifts (Drift).
2.  **Analyzes** drift severity using deep feature extraction (VGG16 backbone).
3.  **Triggers** retraining when drift exceeds critical thresholds (Consecutive failures or high failure rate).
4.  **Validates** the new model against a "Golden Set" to prevent performance decay.
5.  **Deploys** the new model automatically if it passes all checks.

---

## 🚀 Key Features

*   **Intelligent Drift Detection**: Uses a frozen feature extractor (e.g., VGG16) to compute the Wasserstein distance between training data and incoming production data.
*   **Smart Trigger Logic**: automated retraining initiates upon **3 consecutive drift failures** or an **80% failure rate** in the recent window.
*   **Automated Retraining Pipeline**: Seamlessly hands off drifted data to an execution engine (Local/AWS/GCP) to train a challenger model.
*   **Model Decay Gatekeeper**: A "Golden Set" of curvated samples ensures the new model performs better (or at least as good) as the old one before deployment.
*   **Smart Distillation**: Automatically creates lightweight "latent-space" versions of your models for ultra-fast, CPU-friendly drift checking.
*   **Real-time Dashboard**: A comprehensive Streamlit dashboard to visualize system state, drift history, and model performance.
*   **Audit Logging**: Full traceability of every decision, alert, and deployment in a SQLite-backed audit log.

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sentinel.git
    cd sentinel
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Setup Wizard

Sentinel includes an interactive setup wizard to configure the system for your specific environment and data.

Run the setup script:
```bash
python setup.py
```

The wizard will guide you through:
*   **Validating Directories**: ensuring all data and model paths exist.
*   **Drift Calibration**: analyzing your training data to suggest an optimal drift threshold.
*   **Golden Set Creation**: curating a representative set of samples for the gatekeeper check.
*   **Platform Configuration**: choosing between Local, AWS, or GCP for training jobs.

Configuration is saved to `all_config.py`.

---

## 🖥️ Usage

### 1. Start the Sentinel Watcher
The watcher is the core daemon that runs on a schedule (defined in config) to monitor data and orchestrate the pipeline.

```bash
python sentinel_watch.py
```

*   **Drift Check**: Compares incoming data in `data/data_drift/incoming_data` against the baseline.
*   **Retraining**: If triggered, uses `services/execution_engine.py` to train a new model.
*   **Deployment**: If the new model passes the decay check, it replaces the production model in `models/production`.

### 2. Launch the Dashboard
Monitor the system state, view drift charts, and inspect audit logs.

```bash
streamlit run dashboard.py
```
Access the dashboard at `http://localhost:8501`.

---

## 🔧 Configuration

The primary configuration file is `all_config.py`. Key settings include:

*   **`DRIFT_THRESHOLD`**: The Wasserstein distance percentage that indicates significant drift (calibrated during setup).
*   **`RETRAIN_TRIGGER_COUNT`**: Number of consecutive failures to trigger retraining (Default: **3**).
*   **`DRIFT_FAILURE_RATIO`**: Ratio of failures in the recent window to trigger retraining (Default: **0.8** / 80%).
*   **`DECAY_THRESHOLD`**: Max allowed performance drop (%) on the Golden Set for a new model (Default: **5.0%**).
*   **`MONITOR_SCHEDULE`**: CRON expression for how often the watcher runs.

---

## 📂 Directory Structure

```plaintext
sentinel/
├── sentinel_watch.py       # Main monitoring service
├── setup.py                # Interactive setup wizard
├── dashboard.py            # Streamlit dashboard
├── all_config.py           # Central configuration
├── data/
│   ├── data_drift/         # Incoming production data & history
│   ├── golden_set/         # Curated samples for decay checking
│   └── model_decay/        # Logs and history for model performance
├── models/
│   ├── production/         # Active model serving production
│   └── challenger/         # Newly trained models awaiting validation
├── services/               # Core services
│   ├── execution_engine.py # Manages training jobs
│   ├── distiller.py        # Creates latent-space models
│   └── system_state_tracker.py # Tracks global system health
└── tests/                  # Unit tests
```

---

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
