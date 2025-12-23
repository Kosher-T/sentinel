import os
import numpy as np
import sys
import cv2
import keras
from pathlib import Path
import feature_extractor as extractor
from concurrent.futures import ThreadPoolExecutor

# --- OPTIMIZED CPU CONFIGURATION ---
# Since we are skipping the 2.3GB CUDA Toolkit, we force Keras to optimize for CPU math.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
keras.mixed_precision.set_global_policy('float32') # Standard for CPU stability

# --- PROJECT ROOT CALCULATION ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- PATH CONFIGURATION ---
BASE_DATA_DIR = project_root / "data" / "golden_set_septuplets"
OLD_MODEL_PATH = BASE_DATA_DIR / "models" / "old_model" / "vfi_septuplet_epoch_31.keras"
FRESH_MODEL_PATH = BASE_DATA_DIR / "models" / "fresh_model" / "vfi_septuplet_epoch_35.keras"

OLD_RESULTS_DIR = project_root / "data" / "model_decay" / "old_model_results"
FRESH_RESULTS_DIR = project_root / "data" / "model_decay" / "fresh_model_results"
EMBEDDINGS_ROOT = project_root / "data" / "model_decay" / "embeddings"

# --- VFI INFERENCE HELPERS ---

def load_vfi_model(model_path):
    """Loads a VFI model. Note: compile=False avoids needing training optimizers on CPU."""
    print(f"Loading VFI Model: {model_path}")
    return keras.models.load_model(model_path, compile=False)

def prepare_input_sequence(seq_path):
    """Loads im1-im6, resizes to 256x256."""
    frames = []
    original_dims = None
    
    # Load frames 1-6
    for i in range(1, 7):
        img_path = seq_path / f"im{i}.webp"
        if not img_path.exists():
            img_path = seq_path / f"im{i}.png"
            
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None
            
        if original_dims is None:
            h, w = img.shape[:2]
            original_dims = (w, h)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        frames.append(img_resized)
    
    # Concatenate along channel axis (18 channels total)
    input_data = np.concatenate(frames, axis=-1).astype('float32') / 255.0
    return np.expand_dims(input_data, axis=0), original_dims

def run_vfi_inference(vfi_model, output_base_dir, model_name="model"):
    """Runs inference across the golden set sequences."""
    print(f"🎬 Inference Start: {model_name}")
    
    raw_data_path = BASE_DATA_DIR / "sequences" 
    if not raw_data_path.exists():
        print(f"❌ Sequence data missing at {raw_data_path}")
        return

    sequence_dirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir()])
    
    for seq_dir in sequence_dirs:
        seq_id = seq_dir.name
        output_seq_path = output_base_dir / seq_id
        
        # Check if already processed to save CPU time
        if (output_seq_path / "im7_pred.webp").exists():
            continue

        output_seq_path.mkdir(parents=True, exist_ok=True)
        input_tensor, original_dims = prepare_input_sequence(seq_dir)
        
        if input_tensor is None: continue

        # Model Prediction
        preds = vfi_model.predict(input_tensor, verbose=0)
        
        # Extract heads
        im7_raw = (preds[0][0] * 255).astype(np.uint8) 
        im4_raw = (preds[1][0] * 255).astype(np.uint8)

        # Upscale to original size for visual comparison
        im7_final = cv2.resize(im7_raw, original_dims, interpolation=cv2.INTER_CUBIC)
        im4_final = cv2.resize(im4_raw, original_dims, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(str(output_seq_path / "im4_pred.webp"), cv2.cvtColor(im4_final, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_seq_path / "im7_pred.webp"), cv2.cvtColor(im7_final, cv2.COLOR_RGB2BGR))
        
    print(f"✅ Completed inference for {model_name}")

def extract_and_save_embeddings(results_dir, model_id):
    """Generates feature embeddings for the predicted frames."""
    print(f"🚀 Feature Extraction: {model_id}")
    
    feature_model = extractor.create_embedding_model()
    target_emb_dir = EMBEDDINGS_ROOT / model_id
    target_emb_dir.mkdir(parents=True, exist_ok=True)
    
    im4_paths, im7_paths = [], []
    for seq_folder in sorted(results_dir.iterdir()):
        if seq_folder.is_dir():
            p4, p7 = seq_folder / "im4_pred.webp", seq_folder / "im7_pred.webp"
            if p4.exists(): im4_paths.append(str(p4))
            if p7.exists(): im7_paths.append(str(p7))

    for tag, paths in [("im4", im4_paths), ("im7", im7_paths)]:
        if paths:
            # feature_extractor handles batching internally
            emb = extractor.extract_features(feature_model, paths)
            np.save(target_emb_dir / f"{tag}_embeddings.npy", emb)
            print(f"✅ Saved {model_id} {tag} embeddings.")

if __name__ == "__main__":
    # Process fresh baseline if needed
    try:
        FRESH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fresh_model = load_vfi_model(FRESH_MODEL_PATH)
        run_vfi_inference(fresh_model, FRESH_RESULTS_DIR, "Fresh Model")
        extract_and_save_embeddings(FRESH_RESULTS_DIR, "fresh_model")
        del fresh_model
        keras.backend.clear_session()
    except Exception as e:
        print(f"⚠️ Fresh model processing skipped: {e}")

    # Process old model for decay analysis
    try:
        OLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        old_model = load_vfi_model(OLD_MODEL_PATH)
        run_vfi_inference(old_model, OLD_RESULTS_DIR, "Old Model")
        extract_and_save_embeddings(OLD_RESULTS_DIR, "old_model")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n🎉 Model Decay Check Data Generation Complete.")