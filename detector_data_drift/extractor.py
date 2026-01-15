import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import keras

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

def get_data_mode(input_data):
    """
    Analyzes input to determine if we are in 'image' or 'tabular' mode.
    """
    if isinstance(input_data, list):
        if not input_data: return "unknown"
        first_item = input_data[0]
        if isinstance(first_item, list):
            if not first_item: return "unknown"
            first_item = first_item[0]
        ext = Path(first_item).suffix.lower()
    else:
        path = Path(input_data)
        if not path.exists(): return "unknown"
        first_file = next((f for f in path.rglob('*') if f.is_file()), None)
        if not first_file: return "unknown"
        ext = first_file.suffix.lower()
    
    img_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    tab_exts = {'.csv', '.parquet', '.xlsx'}
    
    if ext in img_exts: return "image"
    if ext in tab_exts: return "tabular"
    return "unknown"

# --- NEW CONSOLIDATED LOGIC ---

def get_model_specs(model_path):
    """
    Phase 1: Loads model to extract specifications for the pipeline.
    Returns: (model_object, specs_dict)
    """
    if not model_path or not Path(model_path).exists():
        logging.warning("No valid model path. Returning default specs.")
        return None, {
            "stack_size": 1,
            "target_h": 224,
            "target_w": 224,
            "expected_channels": 3
        }

    try:
        logging.info(f"🟢 Loading model for specification: {Path(model_path).name}")
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        
        shape = model.input_shape
        if isinstance(shape, list): shape = shape[0]
        
        target_h = shape[1] if shape[1] else 224
        target_w = shape[2] if shape[2] else 224
        channels = shape[3] if shape[3] else 3
        
        # Determine stack size (e.g., 18 channels / 3 = 6 images)
        stack_size = 1
        if channels > 3 and channels % 3 == 0:
            stack_size = channels // 3
            
        specs = {
            "stack_size": stack_size,
            "target_h": target_h,
            "target_w": target_w,
            "expected_channels": channels
        }
        
        return model, specs
        
    except Exception as e:
        logging.error(f"🔴 Failed to load model specs: {e}")
        return None, None

# --- IMAGE EXTRACTOR LOGIC ---

def get_recursive_image_paths(directory, extensions=('.png', '.jpg', '.jpeg', '.webp')):
    dir_path = Path(directory)
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(dir_path.rglob(f"*{ext}")))
    return [str(p) for p in image_paths]

def load_and_stack_images(path_group, target_h, target_w):
    """Loads a single image or a stack of images and returns a numpy array."""
    if isinstance(path_group, (str, Path)):
        img = image.load_img(path_group, target_size=(target_h, target_w))
        return image.img_to_array(img)
    
    if isinstance(path_group, list):
        loaded_imgs = []
        for p in path_group:
            img = image.load_img(p, target_size=(target_h, target_w))
            loaded_imgs.append(image.img_to_array(img))
        return np.concatenate(loaded_imgs, axis=-1)
    return None

def extract_image_features(model, image_groups, specs):
    """
    Phase 2: Uses the already loaded model and specs to generate embeddings.
    """
    if not image_groups or model is None:
        logging.error("🔴 Extraction failed: Missing data or model instance.")
        return np.array([]).reshape(0, 0)

    target_h = specs.get("target_h", 224)
    target_w = specs.get("target_w", 224)

    all_embeddings = []
    logging.info(f"📸 Generating embeddings for {len(image_groups)} groups...")
    
    for i in range(0, len(image_groups), BATCH_SIZE):
        batch_groups = image_groups[i:i + BATCH_SIZE]
        batch_data = []
        
        for group in batch_groups:
            try:
                stacked_array = load_and_stack_images(group, target_h, target_w)
                if stacked_array is not None:
                    batch_data.append(stacked_array)
            except Exception:
                continue
        
        if not batch_data:
            continue

        batch_array = np.array(batch_data) / 255.0
        features = model.predict(batch_array, verbose=0)
        all_embeddings.append(features)

    if not all_embeddings:
        return np.array([]).reshape(0, 0)

    return np.vstack(all_embeddings)

# --- UNIVERSAL ENTRY POINT ---

def extract_features(model_instance, input_data, specs=None):
    """
    Refined Entry Point:
    Takes a loaded model_instance and optional specs to perform extraction.
    """
    mode = get_data_mode(input_data)
    
    if mode == "image":
        # Ensure we have specs if we are doing image/stack processing
        if specs is None and model_instance is not None:
             _, specs = get_model_specs(None) # Get defaults
        
        image_groups = input_data if isinstance(input_data, list) else get_recursive_image_paths(input_data)
        return extract_image_features(model_instance, image_groups, specs)
    
    elif mode == "tabular":
        logging.info("Detected Data Mode: TABULAR")
        # Logic for tabular remains the same...
        return np.array([]).reshape(0, 0) # Placeholder
    
    return np.array([]).reshape(0, 0)