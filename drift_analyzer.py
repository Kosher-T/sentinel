import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import os

# --- CONFIGURATION ---
# SENSITIVITY_FACTOR controls how quickly the score jumps to 100%.
# Dropping from 2.0 to 1.5 to provide a slightly more relaxed scoring for the magnitude of drift.
SENSITIVITY_FACTOR = 1.5 

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
    # We must have enough samples (rows) to perform PCA. Min samples = min(len(datasets))
    n_samples_min = min(len(baseline), len(drifted))
    n_features = baseline.shape[1]
    
    # If the number of features is greater than min samples, PCA will fail or be meaningless.
    if n_samples_min <= n_features:
        print(f"WARNING: Too few samples ({n_samples_min}) for full PCA. Capping components.")
        # Target variance might be unreachable, cap components by sample size
        n_components_pca = n_samples_min - 1 if n_samples_min > 1 else 1
    else:
        # Default target variance approach
        n_components_pca = 0.95

    print("Running PCA to reduce noise...")
    try:
        pca = PCA(n_components=n_components_pca)
        pca.fit(baseline)
        baseline_pca = pca.transform(baseline)
        drifted_pca = pca.transform(drifted)
        print(f"PCA reduced features from {n_features} to {baseline_pca.shape[1]}")
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
    # Formula: Score = (1 - e^(-sensitivity * distance)) * 100
    
    drift_score = (1 - np.exp(-SENSITIVITY_FACTOR * avg_wasserstein_dist)) * 100

    print("\n--- Summary of Drift Detection (Wasserstein) ---")
    print(f"Average Raw Distance: {avg_wasserstein_dist:.4f}")
    print(f"Calculated Drift Score: {drift_score:.2f}%")
    
    return drift_score, None

if __name__ == "__main__":
    pass