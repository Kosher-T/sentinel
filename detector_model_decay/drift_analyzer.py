# Receives both baseline and drifted datasets as numpy arrays.
# Uses Wasserstein Distance (Earth Mover's Distance) to quantify drift.
# Reduces dimensionality with PCA before distance calculation.

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import os

# --- CONFIGURATION ---
# SENSITIVITY_FACTOR: Adjusts how "alarmist" the decay score is.
# For VFI models, small visual shifts in embeddings often represent significant quality loss.
SENSITIVITY_FACTOR = 2.5

def calculate_decay_score(fresh_embeddings, old_embeddings):
    """
    Quantifies the distance between a freshly trained model's performance
    and a degraded/older version using Wasserstein Distance.
    
    Returns:
        float: A decay percentage (0% = identical, 100% = completely different distribution).
    """
    
    if fresh_embeddings.shape[1] != old_embeddings.shape[1]:
        raise ValueError("Embedding dimensions do not match between models.")

    print(f"📊 Analyzing Decay across {fresh_embeddings.shape[0]} samples...")

    # --- STEP 1: Dimensionality Reduction ---
    # We use PCA to focus on the most significant visual features (contours, motion blur patterns)
    # and ignore minor pixel noise.
    n_components = min(fresh_embeddings.shape[0], 50) # Use top 50 features or less
    
    pca = PCA(n_components=n_components)
    fresh_pca = pca.fit_transform(fresh_embeddings)
    old_pca = pca.transform(old_embeddings)
    
    print(f"🔹 PCA reduction: {fresh_embeddings.shape[1]} -> {n_components} features.")

    # --- STEP 2: Wasserstein Distance (Earth Mover's Distance) ---
    # We measure the 'cost' of turning the fresh distribution into the old one.
    feature_distances = []
    for i in range(n_components):
        dist = wasserstein_distance(fresh_pca[:, i], old_pca[:, i])
        feature_distances.append(dist)
    
    avg_dist = np.mean(feature_distances)

    # --- STEP 3: Non-linear Mapping to Percent ---
    # Using an exponential growth function so that decay becomes 
    # more apparent as the distance increases.
    decay_percent = (1 - np.exp(-SENSITIVITY_FACTOR * avg_dist)) * 100
    
    return round(float(decay_percent), 2)

def run_analysis_pipeline(embeddings_root):
    """
    Loads saved embeddings and runs the comparison for both 
    Interpolation (im4) and Prediction (im7) heads.
    """
    # Paths for Fresh Model
    fresh_im4 = os.path.join(embeddings_root, "fresh_model", "im4_embeddings.npy")
    fresh_im7 = os.path.join(embeddings_root, "fresh_model", "im7_embeddings.npy")
    
    # Paths for Old Model
    old_im4 = os.path.join(embeddings_root, "old_model", "im4_embeddings.npy")
    old_im7 = os.path.join(embeddings_root, "old_model", "im7_embeddings.npy")

    results = {}

    # Check and analyze im4 (Interpolation)
    if os.path.exists(fresh_im4) and os.path.exists(old_im4):
        f_emb = np.load(fresh_im4)
        o_emb = np.load(old_im4)
        results['interpolation_decay'] = calculate_decay_score(f_emb, o_emb)
    
    # Check and analyze im7 (Prediction)
    if os.path.exists(fresh_im7) and os.path.exists(old_im7):
        f_emb = np.load(fresh_im7)
        o_emb = np.load(old_im7)
        results['prediction_decay'] = calculate_decay_score(f_emb, o_emb)

    return results

if __name__ == "__main__":
    # Local test path logic
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    emb_path = os.path.join(project_root, "data", "model_decay", "embeddings")
    
    if os.path.exists(emb_path):
        scores = run_analysis_pipeline(emb_path)
        print("\n--- FINAL DECAY REPORT ---")
        for task, score in scores.items():
            print(f"🚨 {task.replace('_', ' ').title()}: {score}%")
    else:
        print(f"❌ Embeddings directory not found at {emb_path}")