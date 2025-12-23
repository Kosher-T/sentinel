# Quantifies drift using Wasserstein Distance (Embeddings).
# Quantifies perceptual decay using PSNR and SSIM (Pixels).
# Provides an aggregate health score.

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import cv2
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---
SENSITIVITY_FACTOR = 1.2 # Slightly lowered to reduce "volatility" 

def analyze_drift(baseline, drifted):
    """Calculates Wasserstein Distance (Statistical Drift)."""
    if baseline.shape[1] != drifted.shape[1]:
        raise ValueError("Embedding feature counts do not match!")
        
    n_samples_min = min(len(baseline), len(drifted))
    n_components = min(n_samples_min, 50) 
    
    pca = PCA(n_components=n_components)
    pca.fit(baseline)
    
    baseline_pca = pca.transform(baseline)
    drifted_pca = pca.transform(drifted)

    total_distance = 0.0
    for i in range(baseline_pca.shape[1]):
        total_distance += wasserstein_distance(baseline_pca[:, i], drifted_pca[:, i])

    avg_dist = total_distance / baseline_pca.shape[1]
    # Apply sensitivity
    drift_score = (1 - np.exp(-SENSITIVITY_FACTOR * avg_dist)) * 100
    return drift_score

def calculate_visual_metrics(img1_path, img2_path):
    """Calculates PSNR and SSIM between two images."""
    im1 = cv2.imread(str(img1_path))
    im2 = cv2.imread(str(img2_path))
    if im1 is None or im2 is None: return None, None

    psnr = cv2.PSNR(im1, im2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    score_ssim, _ = ssim(gray1, gray2, full=True)  # type: ignore
    return psnr, score_ssim

def calculate_decay_score(f_emb, o_emb, avg_psnr=None, avg_ssim=None, task="im4"):
    """
    Aggregates statistical drift and visual metrics.
    Adjusts thresholds based on task difficulty (im4 vs im7).
    """
    # 1. Statistical Score
    stat_score = analyze_drift(f_emb, o_emb)
    
    if avg_psnr is None or avg_ssim is None:
        return round(stat_score, 2)

    # 2. Task-Specific Normalization (Calibration)
    # Prediction (im7) naturally has lower PSNR. 
    # We set 'Excellent' benchmarks based on your training logs.
    if task == "im7":
        psnr_benchmark = 35.0  # im7 is okay at 35dB
        ssim_benchmark = 0.95
    else:
        psnr_benchmark = 40.0  # im4 should be 40dB+
        ssim_benchmark = 0.98

    # Calculate individual decay components
    # If psnr > benchmark, decay is 0.
    psnr_decay = max(0, min(100, (psnr_benchmark - avg_psnr) * 4))
    ssim_decay = max(0, min(100, (ssim_benchmark - avg_ssim) * 200))
    
    # 3. Aggregate (Heavily weighted toward Statistics but balanced by task reality)
    # We reduce the weight of Stat Score for Prediction to prevent "Ghosting" artifacts from inflating decay.
    stat_weight = 0.4 if task == "im7" else 0.6
    visual_weight = (1.0 - stat_weight) / 2
    
    aggregate = (stat_score * stat_weight) + (psnr_decay * visual_weight) + (ssim_decay * visual_weight)
    
    return round(aggregate, 2)