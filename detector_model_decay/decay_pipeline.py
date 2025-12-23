# Logs environment details (OS, CPU, Backend) for hardware parity debugging.
# Loads VFI models (old and fresh) to generate interpolated frames on the golden set.
# Extracts and saves feature embeddings for both models to be used for decay analysis.
# Performs Wasserstein Distance analysis to quantify model decay.

# Logs environment details.
# Interactive pipeline for VFI model inference (Old vs Fresh).
# Extracts features and performs Wasserstein Decay Analysis.

import os
import numpy as np
import sys
import cv2
import keras
import platform
import psutil
from pathlib import Path
import feature_extractor as extractor
from drift_analyzer import calculate_decay_score, calculate_visual_metrics
import all_config as config

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
keras.mixed_precision.set_global_policy('float32')

def log_environment():
    print("\n" + "=" * 40)
    print("🖥️  ENVIRONMENT DIAGNOSTICS")
    print("-" * 40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Keras Backend: {keras.backend.backend()}")
    print(f"OpenCV Version: {cv2.__version__}")
    print("=" * 40 + "\n")

def load_vfi_model(model_path):
    print(f"\n🧠 Loading Model: {model_path.name}...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def prepare_input_sequence(seq_path):
    frames = []
    original_dims = None
    for i in range(1, 7):
        img_path = seq_path / f"im{i}.webp"
        if not img_path.exists():
            img_path = seq_path / f"im{i}.png"
        img = cv2.imread(str(img_path))
        if img is None: return None, None
        if original_dims is None:
            h, w = img.shape[:2]
            original_dims = (w, h)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        frames.append(img_resized)
    input_data = np.concatenate(frames, axis=-1).astype('float32') / 255.0
    return np.expand_dims(input_data, axis=0), original_dims

def run_vfi_inference(vfi_model, output_base_dir, model_name="model", force=False):
    print(f"\n🎬 Inference Start: {model_name}")
    print("-" * 30)
    raw_data_path = config.BASE_DATA_DIR / "sequences" 
    if not raw_data_path.exists():
        print(f"❌ Sequence data missing at {raw_data_path}")
        return False

    sequence_dirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir()])
    processed_count = 0
    skipped_count = 0
    
    for seq_dir in sequence_dirs:
        seq_id = seq_dir.name
        output_seq_path = output_base_dir / seq_id
        output_seq_path.mkdir(parents=True, exist_ok=True)
        
        if not force and (output_seq_path / "im7_pred.webp").exists():
            skipped_count += 1
            continue

        input_tensor, original_dims = prepare_input_sequence(seq_dir)
        if input_tensor is None: continue

        preds = vfi_model.predict(input_tensor, verbose=0)
        im7_final = cv2.resize((preds[0][0] * 255).astype(np.uint8), original_dims, interpolation=cv2.INTER_CUBIC)
        im4_final = cv2.resize((preds[1][0] * 255).astype(np.uint8), original_dims, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(str(output_seq_path / "im4_pred.webp"), cv2.cvtColor(im4_final, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_seq_path / "im7_pred.webp"), cv2.cvtColor(im7_final, cv2.COLOR_RGB2BGR))
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"   > Processed {processed_count} sequences...")

    print("-" * 30)
    if processed_count > 0:
        print(f"✅ Completed inference for {model_name} ({processed_count} sequences processed).")
        return True
    return False

def extract_and_save_embeddings(results_dir, model_id, force=False):
    print(f"\n🚀 Feature Extraction: {model_id}")
    print("-" * 30)
    feature_model = extractor.create_embedding_model()
    target_emb_dir = config.EMBEDDINGS_ROOT / model_id
    target_emb_dir.mkdir(parents=True, exist_ok=True)
    
    im4_paths, im7_paths = [], []
    for seq_folder in sorted(results_dir.iterdir()):
        if seq_folder.is_dir():
            p4, p7 = seq_folder / "im4_pred.webp", seq_folder / "im7_pred.webp"
            if p4.exists(): im4_paths.append(str(p4))
            if p7.exists(): im7_paths.append(str(p7))

    tasks_run = 0
    for tag, paths in [("im4", im4_paths), ("im7", im7_paths)]:
        if paths:
            save_file = target_emb_dir / f"{tag}_embeddings.npy"
            if not force and save_file.exists():
                print(f"   ℹ️  {model_id} {tag} embeddings exist. Skipping.")
                continue
            print(f"   > Extracting {tag} features ({len(paths)} frames)...")
            emb = extractor.extract_features(feature_model, paths)
            np.save(save_file, emb)
            tasks_run += 1
    if tasks_run == 0:
        print(f"✅ All embeddings for {model_id} are up to date.")

def run_decay_analysis():
    print("\n" + "=" * 40)
    print("⚖️  STARTING MULTI-METRIC DECAY ANALYSIS")
    print("=" * 40)
    
    tasks = ["im4", "im7"]
    final_report = {}

    for task in tasks:
        fresh_emb = config.EMBEDDINGS_ROOT / "fresh_model" / f"{task}_embeddings.npy"
        old_emb = config.EMBEDDINGS_ROOT / "old_model" / f"{task}_embeddings.npy"
        
        if not (fresh_emb.exists() and old_emb.exists()):
            continue

        print(f"\n📊 Calculating Visual Perceptual Metrics for {task}...")
        psnrs, ssims = [], []
        
        # Iterate through actual results to get pixel-level comparison
        seq_folders = sorted([d for d in config.FRESH_RESULTS_DIR.iterdir() if d.is_dir()])
        for seq_dir in seq_folders:
            fresh_img = seq_dir / f"{task}_pred.webp"
            old_img = config.OLD_RESULTS_DIR / seq_dir.name / f"{task}_pred.webp"
            
            if fresh_img.exists() and old_img.exists():
                p, s = calculate_visual_metrics(fresh_img, old_img)  #type: ignore
                if p is not None:
                    psnrs.append(p)
                    ssims.append(s)

        avg_psnr = np.mean(psnrs) if psnrs else None
        avg_ssim = np.mean(ssims) if ssims else None
        
        print(f"   > Mean PSNR: {avg_psnr:.2f} dB")
        print(f"   > Mean SSIM: {avg_ssim:.4f}")

        # Combine with Statistical (Wasserstein) Drift
        f_emb_data = np.load(fresh_emb)
        o_emb_data = np.load(old_emb)
        final_report[task] = calculate_decay_score(f_emb_data, o_emb_data, avg_psnr, avg_ssim)  #type: ignore

    if final_report:
        print("\n" + "="*40)
        print("📊 AGGREGATE MODEL DECAY REPORT")
        print("="*40)
        for task, score in final_report.items():
            status = "🔴 DEGRADED" if score > 15 else "🟢 STABLE"
            label = "Interpolation" if task == "im4" else "Prediction"
            print(f"{label:<15}: {score:>6}% Decay | {status}")
        print("="*40 + "\n")

if __name__ == "__main__":
    log_environment()
    
    # Process Fresh
    fresh_inf = False
    try:
        config.FRESH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fresh_model = load_vfi_model(config.FRESH_MODEL_PATH)
        if fresh_model:
            fresh_inf = run_vfi_inference(fresh_model, config.FRESH_RESULTS_DIR, "Fresh Model")
            del fresh_model
            keras.backend.clear_session()
    except Exception as e: print(f"⚠️ Fresh error: {e}")
    extract_and_save_embeddings(config.FRESH_RESULTS_DIR, "fresh_model", force=fresh_inf)

    # Process Old
    print("\n" + "-"*40)
    user_choice = input("🔄 Run inference for Old Model? (y/n) [default: y]: ").strip().lower()
    if user_choice != 'n':
        old_inf = False
        try:
            config.OLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            old_model = load_vfi_model(config.OLD_MODEL_PATH)
            if old_model:
                old_inf = run_vfi_inference(old_model, config.OLD_RESULTS_DIR, "Old Model", force=True)
                extract_and_save_embeddings(config.OLD_RESULTS_DIR, "old_model", force=old_inf)
                del old_model
                keras.backend.clear_session()
        except Exception as e: print(f"❌ Old error: {e}")
    else:
        extract_and_save_embeddings(config.OLD_RESULTS_DIR, "old_model", force=False)

    run_decay_analysis()