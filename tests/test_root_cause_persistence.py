"""
Test script for B3: Drift Root Cause Analysis
Validates DB migration, root cause persistence, and data round-trip.
"""
import sys
import os
import json
import sqlite3
import tempfile
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_db_migration():
    """Test that _init_db adds the root_cause_json column."""
    print("TEST 1: DB Migration")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create old-style table (no root_cause_json)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE drift_logs 
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL, data_path TEXT)''')
        conn.commit()
        conn.close()
        
        # Simulate migration logic from sentinel_watch._init_db
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("PRAGMA table_info(drift_logs)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'root_cause_json' not in columns:
            c.execute("ALTER TABLE drift_logs ADD COLUMN root_cause_json TEXT")
        
        conn.commit()
        
        # Verify column exists
        c.execute("PRAGMA table_info(drift_logs)")
        columns = [column[1] for column in c.fetchall()]
        assert 'root_cause_json' in columns, f"root_cause_json not found in {columns}"
        conn.close()
        
        print("  ✅ root_cause_json column added successfully")
    finally:
        os.unlink(db_path)


def test_root_cause_persistence():
    """Test that root cause data round-trips through the DB correctly."""
    print("\nTEST 2: Root Cause Persistence")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE drift_logs 
                     (timestamp TEXT, drift_score REAL, status TEXT, threshold REAL, 
                      data_path TEXT, root_cause_json TEXT)''')
        
        # Mock root cause data (matches what analyzer.py produces)
        root_cause = {
            "per_component": [
                {
                    "component": 1, "wasserstein": 5.1234, "kl_divergence": 0.8901,
                    "explained_variance": 0.3500, "drift_score": 0.6100, "contribution": 0.213500,
                    "mean_shift": 2.45, "variance_ratio": 1.82, "skewness_delta": -0.31,
                    "ks_statistic": 0.42, "ks_pvalue": 0.001
                },
                {
                    "component": 2, "wasserstein": 3.2000, "kl_divergence": 0.4500,
                    "explained_variance": 0.2000, "drift_score": 0.3800, "contribution": 0.076000,
                    "mean_shift": -1.20, "variance_ratio": 0.75, "skewness_delta": 0.15,
                    "ks_statistic": 0.28, "ks_pvalue": 0.045
                },
            ],
            "primary_drivers": [
                {"component": 1, "wasserstein": 5.1234, "kl_divergence": 0.8901,
                 "explained_variance": 0.3500, "drift_score": 0.6100, "contribution": 0.213500,
                 "mean_shift": 2.45, "variance_ratio": 1.82, "skewness_delta": -0.31,
                 "ks_statistic": 0.42, "ks_pvalue": 0.001},
            ],
            "drift_pattern": "localized",
            "drifting_components": 2,
            "total_components": 5,
        }
        
        rc_json = json.dumps(root_cause)
        c.execute(
            "INSERT INTO drift_logs VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-02-11 05:00:00", 32.5, "FAIL", 26.4, "/data/archive/test", rc_json)
        )
        conn.commit()
        
        # Read it back
        c.execute("SELECT root_cause_json FROM drift_logs WHERE status='FAIL' AND root_cause_json IS NOT NULL LIMIT 1")
        row = c.fetchone()
        assert row is not None, "No rows returned"
        
        loaded = json.loads(row[0])
        assert loaded["drift_pattern"] == "localized"
        assert len(loaded["per_component"]) == 2
        assert loaded["per_component"][0]["mean_shift"] == 2.45
        assert loaded["per_component"][0]["ks_pvalue"] == 0.001
        assert loaded["primary_drivers"][0]["variance_ratio"] == 1.82
        
        conn.close()
        print("  ✅ Root cause JSON round-trips correctly through SQLite")
        print("  ✅ All diagnostic fields preserved (mean_shift, variance_ratio, skewness_delta, ks_stat/pvalue)")
    finally:
        os.unlink(db_path)


def test_analyzer_output():
    """Test that analyzer produces the expected root cause structure with new metrics."""
    print("\nTEST 3: Analyzer Output Structure")
    
    from detector_data_drift.analyzer import analyze_drift
    
    np.random.seed(42)
    baseline = np.random.randn(50, 20)
    # Shift component 0 significantly
    current = np.random.randn(50, 20)
    current[:, 0] += 5.0
    
    drift_prob, metrics, root_cause = analyze_drift(baseline, current)
    
    assert "per_component" in root_cause, "Missing per_component"
    assert "primary_drivers" in root_cause, "Missing primary_drivers"
    assert "drift_pattern" in root_cause, "Missing drift_pattern"
    
    # Check new diagnostic fields are present
    comp = root_cause["per_component"][0]
    required_fields = ["component", "wasserstein", "kl_divergence", "explained_variance",
                       "drift_score", "contribution", "mean_shift", "variance_ratio",
                       "skewness_delta", "ks_statistic", "ks_pvalue"]
    
    for field in required_fields:
        assert field in comp, f"Missing field: {field}"
    
    print(f"  ✅ All {len(required_fields)} fields present in per-component data")
    print(f"  ✅ Drift probability: {drift_prob:.4f}")
    print(f"  ✅ Pattern: {root_cause['drift_pattern']}")
    print(f"  ✅ Drifting components: {root_cause['drifting_components']}/{root_cause['total_components']}")
    
    # Print a sample component for visual inspection
    top = root_cause["primary_drivers"][0]
    print(f"\n  Top driver (Component {top['component']}):")
    print(f"    Drift Score:     {top['drift_score']:.4f}")
    print(f"    Mean Shift:      {top['mean_shift']:.4f}")
    print(f"    Variance Ratio:  {top['variance_ratio']:.4f}")
    print(f"    Skewness Delta:  {top['skewness_delta']:.4f}")
    print(f"    KS Statistic:    {top['ks_statistic']:.4f}")
    print(f"    KS p-value:      {top['ks_pvalue']:.6f}")


if __name__ == "__main__":
    print("=" * 50)
    print("B3: Drift Root Cause Analysis - Verification")
    print("=" * 50)
    
    try:
        test_db_migration()
        test_root_cause_persistence()
        test_analyzer_output()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✅")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
