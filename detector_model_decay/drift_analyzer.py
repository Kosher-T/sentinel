import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
SENSITIVITY_FACTOR = 2.5 

def calculate_decay_score(fresh_embeddings, old_embeddings):
    """
    Mathematical Utility: Quantifies the distribution shift between two sets of embeddings.
    Calculates the 'cost' of transforming the fresh distribution into the old one.
    """
    if fresh_embeddings.shape[1] != old_embeddings.shape[1]:
        raise ValueError(f"Feature mismatch: Fresh({fresh_embeddings.shape[1]}) vs Old({old_embeddings.shape[1]})")

    # 1. Dimensionality Reduction (Focus on significant visual features)
    # n_components is limited by the number of samples or a fixed max of 50
    n_samples = min(fresh_embeddings.shape[0], old_embeddings.shape[0])
    n_components = min(n_samples, 50) 
    
    pca = PCA(n_components=n_components)
    fresh_pca = pca.fit_transform(fresh_embeddings)
    old_pca = pca.transform(old_embeddings)
    
    # 2. Wasserstein Distance (Earth Mover's Distance)
    feature_distances = []
    for i in range(n_components):
        dist = wasserstein_distance(fresh_pca[:, i], old_pca[:, i])
        feature_distances.append(dist)
    
    avg_dist = np.mean(feature_distances)

    # 3. Non-linear Mapping to 0-100%
    decay_percent = (1 - np.exp(-SENSITIVITY_FACTOR * avg_dist)) * 100
    
    return round(float(decay_percent), 2)