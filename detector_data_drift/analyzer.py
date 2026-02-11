import numpy as np
from scipy.stats import wasserstein_distance, entropy, ks_2samp, skew
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
    
    Returns:
        drift_probability: float between 0 and 1
        metrics_breakdown: dict of aggregate metric values
        root_cause: dict with per-component breakdown, primary drivers, and drift pattern
    """
    if len(baseline.shape) > 2:
        baseline = baseline.reshape(baseline.shape[0], -1)
        current = current.reshape(current.shape[0], -1)

    # --- STEP 1: Metric Calculation ---
    n_components = min(baseline.shape[0], baseline.shape[1], 10)
    pca = PCA(n_components=n_components)
    
    b_pca = pca.fit_transform(baseline)
    c_pca = pca.transform(current)

    # Per-component metrics (keep individual scores for root cause)
    per_component_wd = [wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]
    per_component_kl = [kl_divergence(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]
    
    # Explained variance ratio from PCA (how important each component is)
    explained_variance = pca.explained_variance_ratio_

    avg_wd = np.mean(per_component_wd)
    avg_kl = np.mean(per_component_kl)

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

    # --- STEP 4: Root Cause Breakdown ---
    # Each component's "contribution" = its drift magnitude weighted by its explained variance
    per_component = []
    for i in range(n_components):
        # Normalize individual WD using same squashing as aggregate
        s_wd_i = 1 - np.exp(-0.05 * per_component_wd[i])
        s_kl_i = 1 - np.exp(-0.2 * per_component_kl[i])
        # Combined drift score for this component
        component_drift = (s_wd_i + s_kl_i) / 2
        # Weight by how much variance this component explains
        contribution = float(component_drift * explained_variance[i])
        
        # --- Diagnostic Metrics (do NOT affect aggregate drift score) ---
        # Mean Shift: signed directional change in component mean
        mean_shift = float(np.mean(c_pca[:, i]) - np.mean(b_pca[:, i]))
        # Variance Ratio: >1 means spreading out, <1 means compressing
        b_var = np.var(b_pca[:, i])
        c_var = np.var(c_pca[:, i])
        variance_ratio = float(c_var / b_var) if b_var > 1e-10 else 1.0
        # Skewness Delta: change in distribution shape/asymmetry
        skewness_delta = float(skew(c_pca[:, i]) - skew(b_pca[:, i]))
        # KS Test: formal statistical test for distribution difference
        ks_stat, ks_p = ks_2samp(b_pca[:, i], c_pca[:, i])
        
        per_component.append({
            "component": i + 1,
            "wasserstein": round(float(per_component_wd[i]), 4),
            "kl_divergence": round(float(per_component_kl[i]), 4),
            "explained_variance": round(float(explained_variance[i]), 4),
            "drift_score": round(float(component_drift), 4),
            "contribution": round(contribution, 6),
            # Diagnostic fields
            "mean_shift": round(mean_shift, 4),
            "variance_ratio": round(variance_ratio, 4),
            "skewness_delta": round(skewness_delta, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
        })
    
    # Sort by contribution (highest first)
    per_component.sort(key=lambda x: x["contribution"], reverse=True)
    
    # Primary drivers: top 3
    primary_drivers = per_component[:3]
    
    # Drift pattern classification
    # Count how many components have meaningful drift (drift_score > 0.1)
    drifting_count = sum(1 for c in per_component if c["drift_score"] > 0.1)
    if drifting_count <= 2:
        drift_pattern = "localized"
    elif drifting_count <= 4:
        drift_pattern = "moderate"
    else:
        drift_pattern = "widespread"
    
    root_cause = {
        "per_component": per_component,
        "primary_drivers": primary_drivers,
        "drift_pattern": drift_pattern,
        "drifting_components": drifting_count,
        "total_components": n_components,
    }

    return drift_probability, metrics_breakdown, root_cause