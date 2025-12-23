# Quantifies drift using Wasserstein Distance (Embeddings).
# Quantifies perceptual decay using PSNR and SSIM (Pixels).
# Provides an aggregate health score.

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import cv2
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---
SENSITIVITY_FACTOR = 1.5 

def analyze_drift(baseline, drifted):
    """
    Calculates Wasserstein Distance (Earth Mover's Distance) on embeddings.
    Returns a normalized score (0-100%).
    """
    if baseline.shape[1] != drifted.shape[1]:
        raise ValueError("Embedding feature counts do not match!")
        
    n_samples_min = min(len(baseline), len(drifted))
    n_features = baseline.shape[1]
    
    # PCA to reduce noise
    n_components = min(n_samples_min, 50) 
    pca = PCA(n_components=n_components)
    pca.fit(baseline)
    
    baseline_pca = pca.transform(baseline)
    drifted_pca = pca.transform(drifted)

    total_distance = 0.0
    for i in range(baseline_pca.shape[1]):
        total_distance += wasserstein_distance(baseline_pca[:, i], drifted_pca[:, i])

    avg_dist = total_distance / baseline_pca.shape[1]
    drift_score = (1 - np.exp(-SENSITIVITY_FACTOR * avg_dist)) * 100
    return drift_score

def calculate_visual_metrics(img1_path, img2_path):
    """
    Calculates PSNR and SSIM between two images.
    img1: Fresh Model Output (Anchor)
    img2: Old Model Output (Target)
    """
    im1 = cv2.imread(str(img1_path))
    im2 = cv2.imread(str(img2_path))
    
    if im1 is None or im2 is None:
        return None, None

    # PSNR
    psnr = cv2.PSNR(im1, im2)
    
    # SSIM (Structural Similarity)
    # Convert to grayscale for faster/standard SSIM
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    score_ssim, _ = ssim(gray1, gray2, full=True)  #type: ignore
    
    return psnr, score_ssim

def calculate_decay_score(f_emb, o_emb, avg_psnr=None, avg_ssim=None):
    """
    Aggregates statistical drift and visual metrics into one score.
    If PSNR/SSIM are provided, they contribute to the final decay percentage.
    """
    # 1. Statistical Score (Wasserstein)
    stat_score = analyze_drift(f_emb, o_emb)
    
    if avg_psnr is None or avg_ssim is None:
        return round(stat_score, 2)

    # 2. Visual Quality Score (Normalized)
    # PSNR: 40+ is excellent, <20 is bad. Let's normalize 20-40 to 0-100 decay.
    # High PSNR = Low Decay.
    psnr_decay = max(0, min(100, (40 - avg_psnr) * 5))
    
    # SSIM: 1.0 is identical, 0.0 is different.
    ssim_decay = (1.0 - avg_ssim) * 100
    
    # 3. Aggregate (Weighted: 50% Stats, 25% PSNR, 25% SSIM)
    aggregate = (stat_score * 0.5) + (psnr_decay * 0.25) + (ssim_decay * 0.25)
    
    return round(aggregate, 2)