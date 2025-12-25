import numpy as np
from scipy.stats import wasserstein_distance, entropy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def kl_divergence(p_samples, q_samples, bins=50):
    """Calculates KL Divergence between two distributions using a histogram approach."""
    # We add a tiny 1% padding to the range to prevent bin-hopping at the edges
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    padding = (max_val - min_val) * 0.01
    
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(min_val - padding, max_val + padding), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(min_val - padding, max_val + padding), density=True)
    
    p_hist += 1e-10
    q_hist += 1e-10
    
    return entropy(p_hist, q_hist)

def mmd_linear(X, Y):
    """Calculates a linear version of Maximum Mean Discrepancy."""
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    return np.sqrt(np.dot(delta, delta.T))

def analyze_drift(baseline, current):
    """
    Combines Wasserstein, KL, Cosine, and MMD into a single normalized drift score (0.0 to 1.0).
    """
    # Fix: Added random_state=42 and svd_solver='full' for absolute determinism
    n_comp = min(16, baseline.shape[0], baseline.shape[1])
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=42)
    
    b_pca = pca.fit_transform(baseline)
    c_pca = pca.transform(current)

    # --- Metric 1: Wasserstein ---
    wd_list = [wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]
    avg_wd = np.mean(wd_list)

    # --- Metric 2: KL Divergence ---
    kl_list = [kl_divergence(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]
    avg_kl = np.mean(kl_list)  # type:ignore

    # --- Metric 3: Cosine Distance (Centroid Comparison) ---
    b_centroid = np.mean(baseline, axis=0).reshape(1, -1)
    c_centroid = np.mean(current, axis=0).reshape(1, -1)
    cos_dist = 1 - cosine_similarity(b_centroid, c_centroid)[0][0]

    # --- Metric 4: Linear MMD ---
    mmd_val = mmd_linear(baseline, current)

    # --- STEP 2: Normalization (The Squashing Functions) ---
    s_wd = 1 - np.exp(-0.1 * avg_wd) 
    s_kl = 1 - np.exp(-0.5 * avg_kl)
    s_cos = 1 - np.exp(-5.0 * cos_dist)
    s_mmd = 1 - np.exp(-0.05 * mmd_val) 

    # Updated weights: Giving Cosine the majority (60%)
    weights = {
        'cosine': 0.60,
        'wasserstein': 0.20,
        'kl': 0.15,
        'mmd': 0.05
    }

    final_score = (
        s_cos * weights['cosine'] +
        s_wd * weights['wasserstein'] +
        s_mmd * weights['mmd'] +
        s_kl * weights['kl']
    )

    metrics_breakdown = {
        'wasserstein': s_wd,
        'mmd': s_mmd,
        'cosine': s_cos,
        'kl': s_kl
    }

    return final_score, metrics_breakdown