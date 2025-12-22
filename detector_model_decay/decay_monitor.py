import os
import numpy as np
import sys
from pathlib import Path
import feature_extractor as extractor

# Fix imports: Ensure the script can find feature_extractor.py and model_config.py 
# if they are in the project root (sentinel/)
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

# --- SYSTEM AGNOSTIC RELATIVE PATHING ---
# We anchor everything to the project root (sentinel/)
BASE_DATA_DIR = project_root / "data" / "golden_set_septuplets"
OUTPUT_EMBEDDINGS_DIR = project_root / "data" / "model_decay" / "embeddings"

def get_vfi_results(root_dir, frame_type="im4_pred.webp"):
    """
    Finds all instances of a specific predicted frame across all sequence folders.
    Paths are built relative to the project root.
    """
    image_paths = []
    # Folder structure: sentinel/data/golden_set_septuplets/[interpolation|prediction]/001/im4_pred.webp
    for sub_type in ["interpolation", "prediction"]:
        search_path = root_dir / sub_type
        if not search_path.exists():
            continue
        
        # Iterate through sequence folders (001, 002, etc.)
        for seq_folder in search_path.iterdir():
            if seq_folder.is_dir():
                full_path = seq_folder / frame_type
                if full_path.exists():
                    image_paths.append(str(full_path))
    
    return sorted(image_paths)

def save_embeddings(data, filename):
    """Utility to ensure directory exists and save numpy array."""
    if not OUTPUT_EMBEDDINGS_DIR.exists():
        OUTPUT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    save_path = OUTPUT_EMBEDDINGS_DIR / filename
    np.save(save_path, data)
    print(f"✅ Saved {len(data)} embeddings to {save_path}")

def run_decay_embedding_extraction():
    """Phase 2.3: Extract embeddings for the predicted frames from the current model."""
    print(f"🚀 Starting Model Decay Feature Extraction...")
    print(f"📂 Root Data Dir: {BASE_DATA_DIR}")

    # 1. Initialize Feature Extractor (MobileNetV2/VGG16 as per model_config.py)
    try:
        model = extractor.create_embedding_model()
    except Exception as e:
        print(f"❌ Failed to initialize feature extractor: {e}")
        return
    
    # 2. Process im4 (Interpolation Results)
    print("🎬 Processing Interpolation Results (im4)...")
    im4_paths = get_vfi_results(BASE_DATA_DIR, "im4_pred.webp")
    if im4_paths:
        im4_embeddings = extractor.extract_features(model, im4_paths)
        save_embeddings(im4_embeddings, "im4_embeddings.npy")
    else:
        print(f"⚠️ No im4_pred.webp files found in {BASE_DATA_DIR}/[interpolation|prediction]")

    # 3. Process im7 (Prediction Results)
    print("🎬 Processing Prediction Results (im7)...")
    im7_paths = get_vfi_results(BASE_DATA_DIR, "im7_pred.webp")
    if im7_paths:
        im7_embeddings = extractor.extract_features(model, im7_paths)
        save_embeddings(im7_embeddings, "im7_embeddings.npy")
    else:
        print(f"⚠️ No im7_pred.webp files found in {BASE_DATA_DIR}/[interpolation|prediction]")

if __name__ == "__main__":
    run_decay_embedding_extraction()