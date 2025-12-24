# Quantifies drift using Wasserstein Distance (Embeddings).
# Quantifies perceptual decay using PSNR and SSIM (Pixels).
# Provides an aggregate health score.

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import cv2
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---
# Lowered significantly. We want the score to be very low if the models are nearly identical.
SENSITIVITY_FACTOR = 0.5 

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
    
    # Score = (1 - e^(-0.5 * dist)) * 100. 
    # This makes small distances result in very small scores.
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
    Calibrated for VFI models with <2% loss variance.
    """
    # 1. Statistical Score
    stat_score = analyze_drift(f_emb, o_emb)
    
    if avg_psnr is None or avg_ssim is None:
        return round(stat_score, 2)

    # 2. Relaxed Benchmarks
    # Based on training logs, im7 at 37dB is "Great", not "Decaying".
    if task == "im7":
        psnr_benchmark = 36.0 
        ssim_benchmark = 0.96
    else:
        psnr_benchmark = 41.0 
        ssim_benchmark = 0.98

    # Calculate individual decay components
    # We reduce the multiplier (was 4 and 200) to be less aggressive.
    psnr_decay = max(0, min(100, (psnr_benchmark - avg_psnr) * 2))
    ssim_decay = max(0, min(100, (ssim_benchmark - avg_ssim) * 50))
    
    # 3. Aggregate
    # For nearly identical models, we trust the Visual Metrics (PSNR/SSIM) more 
    # because Wasserstein is too "jumpy" for fine-tuned weight shifts.
    stat_weight = 0.3 
    visual_weight = 0.35 # (35% PSNR, 35% SSIM)
    
    aggregate = (stat_score * stat_weight) + (psnr_decay * visual_weight) + (ssim_decay * visual_weight)
    
    return round(aggregate, 2)