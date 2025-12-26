import numpy as np
from scipy.stats import wasserstein_distance, entropy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from skimage.metrics import structural_similarity as ssim

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

def analyze_statistical_drift(baseline, current):
    """
    Comprehensive statistical drift analysis for embeddings.
    Returns a normalized score (0.0 to 1.0).
    """
    # Deterministic PCA for consistency
    n_comp = min(16, baseline.shape[0], baseline.shape[1])
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=42)
    
    b_pca = pca.fit_transform(baseline)
    c_pca = pca.transform(current)

    # 1. Wasserstein
    wd_list = [wasserstein_distance(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]
    avg_wd = np.mean(wd_list)

    # 2. KL Divergence
    kl_list = [kl_divergence(b_pca[:, i], c_pca[:, i]) for i in range(b_pca.shape[1])]
    avg_kl = np.mean(kl_list)  # type:ignore

    # 3. Cosine Centroid Distance
    b_centroid = np.mean(baseline, axis=0).reshape(1, -1)
    c_centroid = np.mean(current, axis=0).reshape(1, -1)
    cos_dist = 1 - cosine_similarity(b_centroid, c_centroid)[0][0]

    # 4. MMD
    mmd_val = mmd_linear(baseline, current)

    # Squashing functions (Calibrated for VGG16 embeddings)
    s_wd = 1 - np.exp(-0.1 * avg_wd)
    s_kl = 1 - np.exp(-0.5 * avg_kl)
    s_cos = 1 - np.exp(-5.0 * cos_dist)
    s_mmd = 1 - np.exp(-0.05 * mmd_val)

    # Weighting: Cosine is the strongest signal for 'directional' shift in models
    weights = {'cosine': 0.60,
               'wasserstein': 0.20,
               'kl': 0.15,
               'mmd': 0.05}
    
    return (s_cos * weights['cosine'] + s_wd * weights['wasserstein'] + 
            s_kl * weights['kl'] + s_mmd * weights['mmd'])

def calculate_visual_metrics(im1, im2):
    """Standard PSNR and SSIM calculation."""
    psnr = cv2.PSNR(im1, im2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    score_ssim, _ = ssim(gray1, gray2, full=True)  # type:ignore
    return psnr, score_ssim

def calculate_decay_score(f_emb, o_emb, avg_psnr=None, avg_ssim=None, task="im4"):
    """
    Aggregates statistical embedding drift and visual quality metrics.
    Final output is a percentage (0-100%).
    """
    # 1. Get Statistical Score (0-1 range)
    stat_prob = analyze_statistical_drift(f_emb, o_emb)
    stat_score = stat_prob * 100
    
    if avg_psnr is None or avg_ssim is None:
        return round(stat_score, 2)

    # 2. Perceptual Benchmarks
    if task == "im7":
        psnr_benchmark = 36.0 
        ssim_benchmark = 0.96
    else:
        psnr_benchmark = 41.0 
        ssim_benchmark = 0.98

    # Calculate individual visual decay components
    psnr_decay = max(0, min(100, (psnr_benchmark - avg_psnr) * 2))
    ssim_decay = max(0, min(100, (ssim_benchmark - avg_ssim) * 100))

    # Aggregated Score: 60% Statistical, 40% Perceptual
    # This ensures that even if images 'look' okay, if the logic shifted, we catch it.
    visual_score = (psnr_decay * 0.5) + (ssim_decay * 0.5)
    final_decay = (stat_score * 0.6) + (visual_score * 0.4)

    return round(final_decay, 2)