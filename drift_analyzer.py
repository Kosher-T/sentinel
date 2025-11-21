import numpy as np
from scipy.stats import ks_2samp
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASELINE_FILE = 'baseline_embeddings.npy'
DRIFTED_FILE = 'drifted_embeddings.npy'
P_VALUE_THRESHOLD = 0.05 
# The p-value threshold determines statistical significance. 
# A result where p < 0.05 means there is a strong statistical difference (DRIFT).

def load_embeddings():
    """Loads the baseline and drifted embeddings from the .npy files."""
    try:
        baseline = np.load(BASELINE_FILE)
        drifted = np.load(DRIFTED_FILE)
        
        print(f"Loaded Baseline Embeddings: {baseline.shape}")
        print(f"Loaded Drifted Embeddings: {drifted.shape}")
        
        # Ensure the embeddings have the same number of features (1280 for MobileNetV2)
        if baseline.shape[1] != drifted.shape[1]:
            raise ValueError("Embedding feature counts do not match!")
            
        return baseline, drifted
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Have you run drift_detector.py? {e}")
        return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None

def analyze_drift(baseline, drifted):
    """
    Performs the Kolmogorov-Smirnov (KS) Test feature-by-feature 
    to quantify distribution drift.
    """
    num_features = baseline.shape[1]
    
    # Store results for plotting and threshold setting
    significant_drift_count = 0
    p_values = []
    
    print(f"\n--- Running KS Test on {num_features} features ---")

    # Iterate through each of the 1280 features (columns)
    for i in range(num_features):
        baseline_feature = baseline[:, i] # All values for feature 'i' in baseline data
        drifted_feature = drifted[:, i]   # All values for feature 'i' in drifted data
        
        # ks_2samp returns a statistic (D) and a p-value
        # The p-value tells us the probability that the two distributions are the same.
        statistic, p_value = ks_2samp(baseline_feature, drifted_feature)
        p_values.append(p_value)

        # If p-value < 0.05, we reject the null hypothesis (they are the same)
        # This means the feature has statistically drifted.
        if p_value < P_VALUE_THRESHOLD:
            significant_drift_count += 1
            
    # Calculate the overall "Drift Score"
    drift_score = (significant_drift_count / num_features) * 100
    
    print("\n--- Summary of Drift Detection ---")
    print(f"Total features analyzed: {num_features}")
    print(f"Features showing significant drift (p < {P_VALUE_THRESHOLD}): {significant_drift_count}")
    print(f"Overall Drift Score: {drift_score:.2f}%")
    
    return drift_score, p_values

def plot_drift_histogram(p_values):
    """Visualizes the distribution of p-values to help set a robust threshold."""
    plt.figure(figsize=(10, 5))
    plt.hist(p_values, bins=50, log=True, color='#007acc', edgecolor='black')
    plt.axvline(P_VALUE_THRESHOLD, color='red', linestyle='dashed', linewidth=2, label=f'Drift Threshold (p={P_VALUE_THRESHOLD})')
    plt.title('Distribution of Feature p-values (KS Test)')
    plt.xlabel('P-value')
    plt.ylabel('Number of Features (Log Scale)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('drift_analysis_histogram.png')
    print("\nSaved histogram to 'drift_analysis_histogram.png'")


if __name__ == "__main__":
    baseline, drifted = load_embeddings()
    
    if baseline is not None and drifted is not None:
        drift_score, p_values = analyze_drift(baseline, drifted)
        plot_drift_histogram(p_values)
        
        # This is your final metric to define the production threshold!
        print("\nACTION REQUIRED: Based on this score, set your production alert threshold.")
        print(f"  Since the 'Bad Data' scored {drift_score:.2f}%, you might set the alert trigger at 30-40%.")