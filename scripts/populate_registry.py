import sqlite3
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# Assuming standard path structure
DB_PATH = Path("data/data_drift/model_registry.db")

def populate_registry():
    print(f"Populating Model Registry at {DB_PATH}...")
    
    # Ensure directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Init DB schema if not exists (ModelRegistry class creates it, but we can do it manually or assume existing)
    # We will just use raw SQL to insert
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if table exists, if not create it (simplified from ModelRegistry._init_db)
    c.execute('''CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE NOT NULL,
        registered_at TEXT NOT NULL,
        model_path TEXT,
        source TEXT,
        trigger_reason TEXT,
        parent_version TEXT,
        status TEXT NOT NULL DEFAULT 'registered',
        training_metrics_json TEXT,
        decay_metrics_json TEXT,
        deployment_timestamp TEXT,
        notes TEXT
    )''')
    
    # Clear existing data for clean state
    c.execute("DELETE FROM model_versions")
    
    # Base timestamp
    base_time = datetime.now() - timedelta(days=10)
    
    versions = []
    
    # Generate 12 versions
    for i in range(1, 13):
        version = f"v{i}"
        
        # Determine status and metrics based on requested scenario
        # v1-v9: Mixed history
        # v10: Rejected
        # v11: Deployed
        # v12: Validated (latest candidate)
        
        if i == 12:
            status = "validated"
            source = "Manual"
            trigger = "Manual Retrain"
            acc = 0.981
            loss = 0.02
            score = 0.99
            deployed_ts = None
            reg_time = datetime.now() - timedelta(hours=2)
            
        elif i == 11:
            status = "deployed"
            source = "Retrain"
            trigger = "Drift Detected"
            acc = 0.975
            loss = 0.03
            score = 0.98
            deployed_ts = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')
            reg_time = datetime.now() - timedelta(days=2, hours=4)
            
        elif i == 10:
            status = "rejected"
            source = "Retrain"
            trigger = "Scheduled"
            acc = 0.962
            loss = 0.05
            score = 0.85
            deployed_ts = None
            reg_time = datetime.now() - timedelta(days=5)
            
        else:
            # Random older versions
            status = random.choice(["rejected", "registered", "validated"])
            source = random.choice(["Manual", "Retrain", "Rebase"])
            trigger = "Routine Check"
            acc = 0.90 + (i * 0.005)
            loss = 0.10 - (i * 0.005)
            score = 0.80 + (i * 0.01)
            deployed_ts = None
            reg_time = base_time + timedelta(days=i*0.5)

        # JSON dumps
        train_metrics = json.dumps({"accuracy": acc, "loss": loss, "epochs": 50})
        decay_metrics = json.dumps({"score": score, "ks_statistic": 0.05})
        
        c.execute(
            """INSERT INTO model_versions
               (version, registered_at, model_path, source, trigger_reason,
                parent_version, status, training_metrics_json, decay_metrics_json, deployment_timestamp, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (version, reg_time.strftime('%Y-%m-%d %H:%M:%S'), f"/models/{version}.pkl", source, trigger,
             f"v{i-1}" if i > 1 else None, status, train_metrics, decay_metrics, deployed_ts, "Auto-generated mock data")
        )
        
    conn.commit()
    count = c.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
    conn.close()
    
    print(f"Successfully populated {count} versions into {DB_PATH}")

if __name__ == "__main__":
    populate_registry()
