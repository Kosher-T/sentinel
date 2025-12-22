import os
import numpy as np
import feature_extractor as extractor
from pathlib import Path

# --- RELATIVE PATHING ---
# Assumes script is run from the project root
BASE_DATA_DIR = os.path.join(".", "data", "vfi_golden_set_webp")
OUTPUT_EMBEDDINGS_DIR = os.path.join(".", "data", "model_decay", "embeddings")

def get_vfi_results(root_dir, frame_type="im4_pred.webp"):
    """
    Finds all instances of a specific predicted frame across all sequence folders.
    frame_type options: 'im4_pred.webp' or 'im7_pred.webp'
    """
    image_paths = []
    # Search in both interpolation and prediction subfolders
    for folder in ["interpolation", "prediction"]:
        search_path = os.path.join(root_dir, folder)
        if not os.path.exists(search_path): continue
        
        for seq_folder in os.listdir(search_path):
            full_path = os.path.join(search_path, seq_folder, frame_type)
            if os.path.exists(full_path):
                image_paths.append(full_path)
    
    return sorted(image_paths)

def run_decay_embedding_extraction():
    """Phase 2.3: Extract embeddings for the predicted frames."""
    print("🚀 Starting Model Decay Feature Extraction...")
    
    if not os.path.exists(OUTPUT_EMBEDDINGS_DIR):
        os.makedirs(OUTPUT_EMBEDDINGS_DIR)

    # 1. Initialize Model
    model = extractor.create_embedding_model()
    
    # 2. Process im4 (Interpolation)
    print("🎬 Processing Interpolation Results (im4)...")
    im4_paths = get_vfi_results(BASE_DATA_DIR, "im4_pred.webp")
    if im4_paths:
        im4_embeddings = extractor.extract_features(model, im4_paths)
        save_path = os.path.join(OUTPUT_EMBEDDINGS_DIR, "im4_embeddings.npy")
        np.save(save_path, im4_embeddings)
        print(f"✅ Saved {len(im4_embeddings)} im4 embeddings to {save_path}")
    else:
        print("❌ No im4_pred.webp files found.")

    # 3. Process im7 (Prediction)
    print("🎬 Processing Prediction Results (im7)...")
    im7_paths = get_vfi_results(BASE_DATA_DIR, "im7_pred.webp")
    if im7_paths:
        im7_embeddings = extractor.extract_features(model, im7_paths)
        save_path = os.path.join(OUTPUT_EMBEDDINGS_DIR, "im7_embeddings.npy")
        np.save(save_path, im7_embeddings)
        print(f"✅ Saved {len(im7_embeddings)} im7 embeddings to {save_path}")
    else:
        print("❌ No im7_pred.webp files found.")

if __name__ == "__main__":
    run_decay_embedding_extraction()