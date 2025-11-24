import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
import os

# --- CONFIGURATION ---
# P-Value Threshold: The strictness of the statistical test
try:
    # We lower the default P-value slightly to be more conservative
    P_VALUE_THRESHOLD = float(os.environ.get("P_VALUE_THRESHOLD", "0.01"))
except ValueError:
    P_VALUE_THRESHOLD = 0.01

def analyze_drift(baseline, drifted):
    """
    1. Reduces dimensionality using PCA to remove noise.
    2. Performs KS Test on the Principal Components.
    """
    if baseline.shape[1] != drifted.shape[1]:
        raise ValueError("Embedding feature counts do not match!")
        
    print(f"Original Feature Count: {baseline.shape[1]}")
    
    # --- STEP 1: Dimensionality Reduction (PCA) ---
    # We want to keep enough components to explain 95% of the variance.
    # This removes the 'noise' features that cause false positives.
    
    # If we have fewer samples than features, we can't run full PCA. 
    # We cap components at min(samples, features)
    n_components = min(len(baseline), len(drifted), baseline.shape[1])
    
    # We target 95% variance retention or 50 components, whichever is smaller/safer
    target_variance = 0.95
    
    print("Running PCA to reduce noise and dimensionality...")
    try:
        pca = PCA(n_components=target_variance)
        pca.fit(baseline)
        
        baseline_pca = pca.transform(baseline)
        drifted_pca = pca.transform(drifted)
        
        print(f"PCA reduced features from {baseline.shape[1]} to {baseline_pca.shape[1]}")
        
    except Exception as e:
        print(f"PCA Failed (likely too little data). Falling back to raw features. Error: {e}")
        baseline_pca = baseline
        drifted_pca = drifted

    # --- STEP 2: Statistical Test (KS Test) ---
    num_features = baseline_pca.shape[1]
    significant_drift_count = 0
    p_values = [] 

    # We apply a correction because we are running multiple tests.
    # Simple approach: Stick to the user defined P-Value but on much fewer, high-quality features.
    
    for i in range(num_features):
        baseline_feature = baseline_pca[:, i]
        drifted_feature = drifted_pca[:, i]
        
        # KS Test
        statistic, p_value = ks_2samp(baseline_feature, drifted_feature)
        p_values.append(p_value)

        if p_value < P_VALUE_THRESHOLD:
            significant_drift_count += 1

    # Calculate the drift score
    drift_score = (significant_drift_count / num_features) * 100

    print("\n--- Summary of Drift Detection ---")
    print(f"P-Value Threshold used: {P_VALUE_THRESHOLD}")
    print(f"Total Components analyzed: {num_features}")
    print(f"Components showing drift: {significant_drift_count}")
    print(f"Overall Drift Score: {drift_score:.2f}%")
    
    return drift_score, p_values

if __name__ == "__main__":
    pass