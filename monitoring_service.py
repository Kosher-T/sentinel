import os
import sys
import numpy as np
import drift_detector as detector
import drift_analyzer as analyzer

# --- CONFIGURATION ---
# In the container, we will mount the new data to this path
NEW_DATA_PATH = "/app/incoming_data"
BASELINE_PATH = "/embeddings/baseline_embeddings.npy"
DRIFT_THRESHOLD = 30.0  # Set this based on your histogram (e.g., 30-40%)

def check_for_drift():
    print("--- STARTING MONITORING JOB ---")
    
    # 1. Load the "Memory of Perfection" (Baseline)
    if not os.path.exists(BASELINE_PATH):
        print(f"CRITICAL ERROR: Baseline file {BASELINE_PATH} not found inside container.")
        sys.exit(1)
    
    baseline = np.load(BASELINE_PATH)
    print(f"1. Loaded Baseline: {baseline.shape[0]} frames.")

    # 2. Check if there is new data to inspect
    if not os.path.exists(NEW_DATA_PATH) or not os.listdir(NEW_DATA_PATH):
        print(f"Error: No data found at {NEW_DATA_PATH}. Did you mount the volume?")
        sys.exit(1)

    # 3. The QC "Touch Test" (Generate Embeddings for new data)
    print("2. Loading AI Model (QC Sensor)...")
    model = detector.create_embedding_model()
    
    print(f"3. Inspecting images in {NEW_DATA_PATH}...")
    new_embeddings = detector.generate_embeddings_from_directory(model, NEW_DATA_PATH)
    
    if new_embeddings is None or new_embeddings.size == 0:
        print("Error: Could not generate embeddings from new data.")
        sys.exit(1)

    # 4. The QC "Judgment" (Compare New vs. Baseline)
    print("4. Running Statistical Analysis (QC Judge)...")
    score, _ = analyzer.analyze_drift(baseline, new_embeddings)
    
    print(f"\n>>> DRIFT SCORE: {score:.2f}%")
    print(f">>> THRESHOLD:   {DRIFT_THRESHOLD}%")

    # 5. The Verdict
    if score > DRIFT_THRESHOLD:
        print("\n[FAIL] HIGH DRIFT DETECTED! Triggers required.")
        # In a real app, you might write this to a JSON file for GitHub Actions to read
        with open("/app/status.txt", "w") as f:
            f.write("FAIL")
        sys.exit(1) # Exit with error code to signal failure to Docker/GitHub
    else:
        print("\n[PASS] Model is operating within normal parameters.")
        with open("/app/status.txt", "w") as f:
            f.write("PASS")
        sys.exit(0) # Exit with success code

if __name__ == "__main__":
    check_for_drift()