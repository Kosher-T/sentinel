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
    
    p_hist += 1e-10
    q_hist += 1e-10
    
    return entropy(p_hist, q_hist)

def mmd_linear(X, Y):
    """Calculates a linear version of Maximum Mean Discrepancy."""
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    return np.sqrt(np.dot(delta, delta.T))

def analyze_drift(baseline, current):
    """
    Compares two embedding distributions. 
    Adjusted to be less sensitive to magnitude shifts and trust Cosine more.
    """
    if len(baseline.shape) > 2:
        baseline = baseline.reshape(baseline.shape[0], -1)
        current = current.reshape(current.shape[0], -1)

    # --- STEP 1: Metric Calculation ---
    n_components = min(baseline.shape[0], baseline.shape[1], 10)
    pca = PCA(n_components=n_components)
    
    b_pca = pca.fit_transform(baseline)
    c_pca = pca.transform(current)

    avg_wd = np.mean([wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])])
    avg_kl = np.mean([kl_divergence(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]) # type:ignore

    b_centroid = np.mean(baseline, axis=0).reshape(1, -1)
    c_centroid = np.mean(current, axis=0).reshape(1, -1)
    cos_dist = 1 - cosine_similarity(b_centroid, c_centroid)[0][0]

    mmd_val = mmd_linear(baseline, current)

    # --- STEP 2: Normalization (The Squashing Functions) ---
    # Coefficients adjusted to handle larger raw values without hitting 1.0 immediately
    s_wd = 1 - np.exp(-0.05 * avg_wd)   # Was -0.2
    s_kl = 1 - np.exp(-0.2 * avg_kl)    # Was -0.5
    s_cos = 1 - np.exp(-10.0 * cos_dist) # Increased sensitivity to capture small changes
    s_mmd = 1 - np.exp(-0.01 * mmd_val)

    # --- STEP 3: Weighted Ensembling (Cosine Majority) ---
    weights = {
        "Wasserstein": 0.04,
        "KL_Div": 0.05,
        "Cosine_Centroid": 0.90,
        "Linear_MMD": 0.01
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