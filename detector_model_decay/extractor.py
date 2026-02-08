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
    from keras.preprocessing import image # type:ignore

# Load global configuration
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

# Configure local logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] DECAY_EXTRACTOR: %(message)s')

BATCH_SIZE = 32

def create_embedding_model():
    """
    Creates a VGG16-based feature extraction model.
    Returns a Keras model that outputs embeddings.
    """
    try:
        from tensorflow.keras.applications import VGG16  # type: ignore
        from tensorflow.keras.layers import GlobalAveragePooling2D  # type: ignore
        from tensorflow.keras.models import Model  # type: ignore
    except ImportError:
        from keras.applications import VGG16  # type: ignore
        from keras.layers import GlobalAveragePooling2D  # type: ignore
        from keras.models import Model  # type: ignore
    
    try:
        target_h, target_w, channels = config.EMBEDDING_INPUT_SHAPE
    except (AttributeError, ValueError):
        target_h, target_w, channels = 224, 224, 3
    
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(target_h, target_w, channels)
    )
    
    # Add global average pooling to get fixed-size embeddings
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    logging.info(f"✅ Created VGG16 embedding model with output shape: {model.output_shape}")
    return model

def get_data_mode(directory):
    """
    Analyzes Golden Set contents to determine data type.
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

# --- IMAGE LOGIC ---

def get_recursive_image_paths(directory, extensions=('.png', '.jpg', '.jpeg', '.webp')):
    dir_path = Path(directory)
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(dir_path.rglob(f"*{ext}")))
    return [str(p) for p in image_paths]

def extract_image_features(model, image_paths):
    """Interrogates model using Golden Set images."""
    if not image_paths:
        return np.array([])

    all_embeddings = []
    try:
        target_h, target_w = config.EMBEDDING_INPUT_SHAPE[:2]
    except AttributeError:
        target_h, target_w = 224, 224

    logging.info(f"Generating Golden Set embeddings ({len(image_paths)} images)...")
    
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

        batch_array = np.array(batch_imgs) / 255.0
        features = model.predict(batch_array, verbose=0)
        all_embeddings.append(features)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])

# --- TABULAR LOGIC ---

def extract_tabular_features(directory):
    """Loads Golden Set tabular data."""
    path = Path(directory)
    all_dfs = []
    
    logging.info(f"Loading Golden Set tabular data from {directory}...")
    
    for file in path.rglob('*'):
        try:
            if file.suffix == '.csv':
                all_dfs.append(pd.read_csv(file))
            elif file.suffix == '.parquet':
                all_dfs.append(pd.read_parquet(file))
        except Exception:
            continue

    if not all_dfs:
        return np.array([])

    combined_df = pd.concat(all_dfs, axis=0)
    numeric_df = combined_df.select_dtypes(include=[np.number])
    return numeric_df.values

# --- UNIVERSAL ENTRY POINT ---

def extract_features(model, directory):
    """
    Universal entry point for Decay Check.
    Ensures the 'Gatekeeper' logic works for any data type.
    """
    mode = get_data_mode(directory)
    
    if mode == "image":
        image_paths = get_recursive_image_paths(directory)
        return extract_image_features(model, image_paths)
    
    elif mode == "tabular":
        return extract_tabular_features(directory)
    
    else:
        logging.error(f"🔴 Golden Set format in {directory} is unsupported.")
        return np.array([])

def extract_features_from_array(model, data_array):
    """
    Extracts embeddings directly from an in-memory numpy array.
    Useful for processing stored Golden Set predictions.
    """
    if data_array is None or len(data_array) == 0:
        return np.array([])

    all_embeddings = []
    
    # Check if we need to resize
    try:
        target_h, target_w = config.EMBEDDING_INPUT_SHAPE[:2]
    except AttributeError:
        target_h, target_w = 224, 224

    logging.info(f"Generating embeddings from array shape {data_array.shape}...")

    for i in range(0, len(data_array), BATCH_SIZE):
        batch_data = data_array[i:i + BATCH_SIZE]
        
        # Ensure float32 and normalized [0, 1] if valid image range [0, 255]
        if batch_data.max() > 1.0:
            batch_data = batch_data.astype(np.float32) / 255.0
            
        try:
            features = model.predict(batch_data, verbose=0)
            all_embeddings.append(features)
        except Exception as e:
            logging.error(f"Prediction failed on batch: {e}")
            continue

    return np.vstack(all_embeddings) if all_embeddings else np.array([])