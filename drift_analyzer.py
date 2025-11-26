import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import os

# --- CONFIGURATION ---
# SENSITIVITY_FACTOR controls how quickly the score jumps to 100%.
# Lowering this factor dampens the result, ensuring only large, meaningful raw distances 
# translate into high percentage scores. We reduce it from 10.0 to 2.0.
SENSITIVITY_FACTOR = 2.0

def analyze_drift(baseline, drifted):
    """
    1. Reduces dimensionality using PCA.
    2. Calculates Wasserstein Distance (Earth Mover's Distance).
    3. Maps distance to a 0-100% score using a sigmoid-like function.
    """
    if baseline.shape[1] != drifted.shape[1]:
        raise ValueError("Embedding feature counts do not match!")
        
    print(f"Original Feature Count: {baseline.shape[1]}")
    
    # --- STEP 1: Dimensionality Reduction (PCA) ---
    n_components = min(len(baseline), len(drifted), baseline.shape[1])
    target_variance = 0.95
    
    print("Running PCA to reduce noise...")
    try:
        pca = PCA(n_components=target_variance)
        pca.fit(baseline)
        baseline_pca = pca.transform(baseline)
        drifted_pca = pca.transform(drifted)
        print(f"PCA reduced features from {baseline.shape[1]} to {baseline_pca.shape[1]}")
    except Exception as e:
        print(f"PCA Failed. Falling back to raw features. Error: {e}")
        baseline_pca = baseline
        drifted_pca = drifted

    # --- STEP 2: Wasserstein Distance ---
    num_features = baseline_pca.shape[1]
    total_distance = 0.0

    # We calculate the distance for every feature and average it
    for i in range(num_features):
        b_feat = baseline_pca[:, i]
        d_feat = drifted_pca[:, i]
        
        # Calculate Earth Mover's Distance for this feature
        wd = wasserstein_distance(b_feat, d_feat)
        total_distance += wd

    avg_wasserstein_dist = total_distance / num_features

    # --- STEP 3: Normalize to 0-100% Scale ---
    # We use a simple exponential decay function to map distance (0 to inf) to percentage (0 to 100)
    # Formula: Score = (1 - e^(-sensitivity * distance)) * 100
    # This ensures 0 distance = 0% score, and high distance approaches 100% smoothly.
    
    drift_score = (1 - np.exp(-SENSITIVITY_FACTOR * avg_wasserstein_dist)) * 100

    print("\n--- Summary of Drift Detection (Wasserstein) ---")
    print(f"Average Raw Distance: {avg_wasserstein_dist:.4f}")
    print(f"Calculated Drift Score: {drift_score:.2f}%")
    
    return drift_score, None

if __name__ == "__main__":
    pass