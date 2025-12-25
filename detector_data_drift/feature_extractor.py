import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from pathlib import Path

# Load global configuration
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

# Factory for standard architectures
MODEL_FACTORY = {
    "MobileNetV2": {"class": MobileNetV2, "preprocess": mobile_preprocess},
    "VGG16": {"class": VGG16, "preprocess": vgg_preprocess},
    "ResNet50": {"class": ResNet50, "preprocess": resnet_preprocess}
}

BATCH_SIZE = 32

def get_recursive_image_paths(directory, extensions=('.png', '.jpg', '.jpeg', '.webp')):
    """
    Scours a directory and all subdirectories for images.
    """
    dir_path = Path(directory)
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(dir_path.rglob(f"*{ext}")))
    
    print(f"🔍 Scoured {directory}: Found {len(image_paths)} images across all subfolders.")
    return [str(p) for p in image_paths]

def create_embedding_model():
    """
    Creates the feature extraction model based on config settings.
    """
    if config.EMBEDDING_MODEL_TYPE not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model type: {config.EMBEDDING_MODEL_TYPE}")
    
    base_class = MODEL_FACTORY[config.EMBEDDING_MODEL_TYPE]["class"]
    base = base_class(
        weights='imagenet', 
        include_top=False, 
        input_shape=config.EMBEDDING_INPUT_SHAPE,
        pooling='avg'
    )
    return base

def extract_features_from_list(model, image_paths):
    """
    Generates embeddings from a specific list of image paths.
    This avoids redundant directory scouring.
    """
    if not image_paths:
        return np.array([])

    all_embeddings = []
    target_h, target_w = config.EMBEDDING_INPUT_SHAPE[:2]
    preprocess_func = MODEL_FACTORY[config.EMBEDDING_MODEL_TYPE]["preprocess"]

    print(f"🚀 Generating embeddings for {len(image_paths)} images...")
    
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
        preprocessed_batch = preprocess_func(batch_array)
        
        features = model.predict(preprocessed_batch, verbose=0)
        all_embeddings.append(features)

    if not all_embeddings:
        return np.array([])
        
    return np.vstack(all_embeddings)

def extract_features(model, directory):
    """
    Legacy/Convenience wrapper that discovers files and extracts features.
    """
    image_paths = get_recursive_image_paths(directory)
    return extract_features_from_list(model, image_paths)