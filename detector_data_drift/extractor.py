import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Handle Keras/TF Import Discrepancies
try:
    from tensorflow.keras.preprocessing import image # type:ignore
except ImportError:
    from keras.preprocessing import image

# Load global configuration
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

# Configure local logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] EXTRACTOR: %(message)s')

BATCH_SIZE = 32

def get_data_mode(directory):
    """
    Analyzes folder contents to determine if we are in 'image' or 'tabular' mode.
    """
    path = Path(directory)
    extensions = [f.suffix.lower() for f in path.rglob('*') if f.is_file()]
    
    img_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    tab_exts = {'.csv', '.parquet', '.xlsx'}
    
    if any(ext in img_exts for ext in extensions):
        return "image"
    if any(ext in tab_exts for ext in extensions):
        return "tabular"
    
    return "unknown"

# --- IMAGE EXTRACTOR LOGIC ---

def get_recursive_image_paths(directory, extensions=('.png', '.jpg', '.jpeg', '.webp')):
    dir_path = Path(directory)
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(dir_path.rglob(f"*{ext}")))
    return [str(p) for p in image_paths]

def extract_image_features(model, image_paths):
    """Generates embeddings from images using the distilled/provided model."""
    if not image_paths:
        return np.array([])

    all_embeddings = []
    # Try to infer input shape from config or model
    try:
        target_h, target_w = config.EMBEDDING_INPUT_SHAPE[:2]
    except AttributeError:
        target_h, target_w = 224, 224 # Fallback

    logging.info(f"Generating embeddings for {len(image_paths)} images...")
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_imgs = []
        
        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=(target_h, target_w))
                img_array = image.img_to_array(img)
                batch_imgs.append(img_array)
            except Exception:
                continue
        
        if not batch_imgs:
            continue

        batch_array = np.array(batch_imgs)
        # We assume pre-processing is handled inside the distilled model 
        # or normalized to 0-1 here for general purposes
        batch_array = batch_array / 255.0
        
        features = model.predict(batch_array, verbose=0)
        all_embeddings.append(features)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])

# --- TABULAR EXTRACTOR LOGIC ---

def extract_tabular_features(directory):
    """
    Loads tabular data and ensures it's returned as a standardized 
    NumPy array for drift analysis.
    """
    path = Path(directory)
    all_dfs = []
    
    logging.info(f"Loading tabular data from {directory}...")
    
    for file in path.rglob('*'):
        try:
            if file.suffix == '.csv':
                all_dfs.append(pd.read_csv(file))
            elif file.suffix == '.parquet':
                all_dfs.append(pd.read_parquet(file))
        except Exception as e:
            logging.warning(f"   -> Could not read {file.name}: {e}")

    if not all_dfs:
        return np.array([])

    combined_df = pd.concat(all_dfs, axis=0)
    # Drop non-numeric columns to ensure distance math works
    numeric_df = combined_df.select_dtypes(include=[np.number])
    
    return numeric_df.values

# --- UNIVERSAL ENTRY POINT ---

def extract_features(model, directory):
    """
    Universal entry point that detects data type and returns embeddings/vectors.
    """
    mode = get_data_mode(directory)
    
    if mode == "image":
        logging.info("Detected Data Mode: IMAGE")
        image_paths = get_recursive_image_paths(directory)
        return extract_image_features(model, image_paths)
    
    elif mode == "tabular":
        logging.info("Detected Data Mode: TABULAR")
        return extract_tabular_features(directory)
    
    else:
        logging.error(f"🔴 Unsupported or empty data format in {directory}")
        return np.array([])