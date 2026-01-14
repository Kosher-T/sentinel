import numpy as np
from scipy.stats import wasserstein_distance, entropy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def kl_divergence(p_samples, q_samples, bins=50):
    """Calculates KL Divergence between two distributions using a histogram approach."""
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    padding = (max_val - min_val) * 0.01
    
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(min_val - padding, max_val + padding), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(min_val - padding, max_val + padding), density=True)
    
    # Laplace smoothing to avoid log(0)
    p_hist += 1e-10
    q_hist += 1e-10
    
    return entropy(p_hist, q_hist)

def mmd_linear(X, Y):
    """Calculates a linear version of Maximum Mean Discrepancy."""
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    return np.sqrt(np.dot(delta, delta.T))

def analyze_drift(baseline, current):
    """
    Compares two embedding distributions across multiple dimensions.
    """
    # 1. Flatten if necessary (in case model output is [Batch, H, W, C])
    if len(baseline.shape) > 2:
        baseline = baseline.reshape(baseline.shape[0], -1)
        current = current.reshape(current.shape[0], -1)

    # --- STEP 1: Dimensionality Reduction ---
    # We use PCA to focus on the top components of variance
    n_components = min(baseline.shape[0], baseline.shape[1], 10)
    pca = PCA(n_components=n_components)
    
    b_pca = pca.fit_transform(baseline)
    c_pca = pca.transform(current)

    # --- Metric 1: Wasserstein (Earth Mover's Distance) ---
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
    # These map raw metrics to a 0-1 range based on empirical sensitivity
    s_wd = 1 - np.exp(-0.2 * avg_wd) 
    s_kl = 1 - np.exp(-0.5 * avg_kl)
    s_cos = 1 - np.exp(-5.0 * cos_dist)
    s_mmd = 1 - np.exp(-1.0 * mmd_val)

    # --- STEP 3: Weighted Ensembling ---
    # We prioritize Cosine and MMD as they are more robust to small sample sizes
    weights = {
        "Wasserstein": 0.20,
        "KL_Div": 0.20,
        "Cosine_Centroid": 0.35,
        "Linear_MMD": 0.25
    }
    
    drift_probability = (
        s_wd * weights["Wasserstein"] +
        s_kl * weights["KL_Div"] +
        s_cos * weights["Cosine_Centroid"] +
        s_mmd * weights["Linear_MMD"]
    )

    metrics_breakdown = {
        "Avg Wasserstein": avg_wd,
        "Avg KL Div": avg_kl,
        "Cosine Distance": cos_dist,
        "Linear MMD": mmd_val
    }

    return drift_probability, metrics_breakdown