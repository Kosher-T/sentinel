import numpy as np
from scipy.stats import wasserstein_distance, entropy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def kl_divergence(p_samples, q_samples, bins=50):
    """Calculates KL Divergence between two distributions using a histogram approach."""
    # Create a common range for bins
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(min_val, max_val), density=True)
    
    # Add small epsilon to avoid division by zero or log(0)
    p_hist += 1e-10
    q_hist += 1e-10
    
    return entropy(p_hist, q_hist)

def mmd_linear(X, Y):
    """Calculates a linear version of Maximum Mean Discrepancy."""
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    return np.sqrt(np.dot(delta, delta.T))

def analyze_drift(baseline, current):
    """
    Combines Wasserstein, KL, Cosine, and MMD into a single Drift Score.
    
    Args:
        baseline (np.ndarray): Baseline embeddings (N, D)
        current (np.ndarray): New batch embeddings (M, D)
    Returns:
        float: Final Drift Score (0-100%)
        dict: Breakdown of individual metrics
    """
    if baseline.shape[1] != current.shape[1]:
        raise ValueError("Feature dimensions must match.")

    # 1. Dimensionality Reduction (PCA) 
    # Helps metrics like KL and Wasserstein run faster and on cleaner signals
    n_components = min(0.95, min(baseline.shape[0], current.shape[0]) - 1)
    pca = PCA(n_components=n_components)
    b_pca = pca.fit_transform(baseline)
    c_pca = pca.transform(current)
    num_dims = b_pca.shape[1]

    # --- Metric 1: Wasserstein (EMD) ---
    wd_list = [wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(num_dims)]
    avg_wd = np.mean(wd_list)

    # --- Metric 2: KL Divergence ---
    kl_list = [kl_divergence(b_pca[:, i], c_pca[:, i]) for i in range(num_dims)]
    avg_kl = np.mean(kl_list)

    # --- Metric 3: Cosine Similarity (Centroid Drift) ---
    b_centroid = np.mean(baseline, axis=0).reshape(1, -1)
    c_centroid = np.mean(current, axis=0).reshape(1, -1)
    # Cosine distance = 1 - similarity
    cos_dist = 1 - cosine_similarity(b_centroid, c_centroid)[0][0]

    # --- Metric 4: Linear MMD ---
    mmd_val = mmd_linear(baseline, current)

    # --- STEP 2: Normalization & Weighting ---
    # Every metric has a different natural scale. We map them to 0.0 - 1.0
    # Values based on empirical observation of ResNet/MobileNet embeddings
    s_wd = 1 - np.exp(-1.5 * avg_wd)
    s_kl = 1 - np.exp(-0.5 * avg_kl)
    s_cos = 1 - np.exp(-5.0 * cos_dist)
    s_mmd = 1 - np.exp(-1.2 * mmd_val)

    # Weighted Average (Adjust weights based on what you trust most)
    # Wasserstein and MMD are usually the most stable for images.
    weights = {
        'wasserstein': 0.35,
        'mmd': 0.35,
        'cosine': 0.15,
        'kl': 0.15
    }

    final_score = (
        s_wd * weights['wasserstein'] +
        s_mmd * weights['mmd'] +
        s_cos * weights['cosine'] +
        s_kl * weights['kl']
    ) * 100

    metrics_breakdown = {
        "wasserstein": round(s_wd * 100, 2),
        "mmd": round(s_mmd * 100, 2),
        "cosine": round(s_cos * 100, 2),
        "kl": round(s_kl * 100, 2)
    }

    return round(final_score, 2), metrics_breakdown

if __name__ == "__main__":
    print("Drift Analyzer logic ready. Call analyze_drift(baseline, current) from your service.")