import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess
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
    Returns a list of strings.
    """
    dir_path = Path(directory)
    image_paths = []
    for ext in extensions:
        image_paths.extend([str(p) for p in dir_path.rglob(f"*{ext}")])
    return sorted(image_paths)

def create_embedding_model():
    """Creates a feature extraction model based on all_config.py."""
    cfg_type = config.EMBEDDING_MODEL_TYPE
    if cfg_type not in MODEL_FACTORY:
        raise ValueError(f"Model {cfg_type} not supported.")
    
    base = MODEL_FACTORY[cfg_type]["class"](
        weights='imagenet', 
        include_top=False, 
        input_shape=config.EMBEDDING_INPUT_SHAPE,
        pooling='avg'
    )
    return base

def extract_features(model, input_data):
    """
    Processes either a directory path or a list of image paths.
    Generates embeddings and returns a numpy array.
    """
    # If input is a path/string directory, get the images inside
    if isinstance(input_data, (str, Path)) and Path(input_data).is_dir():
        image_paths = get_recursive_image_paths(input_data)
    elif isinstance(input_data, list):
        image_paths = [str(p) for p in input_data]
    else:
        # Fallback for single Path object that is a file
        image_paths = [str(input_data)]

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
            except Exception as e:
                # Silently skip corrupted images
                continue
        
        if not batch_imgs:
            continue

        # Convert to batch tensor and preprocess
        batch_tensor = np.array(batch_imgs)
        batch_preprocessed = preprocess_func(batch_tensor)
        
        # Inference
        features = model.predict(batch_preprocessed, verbose=0)
        all_embeddings.append(features)

    if not all_embeddings:
        return np.array([])

    return np.vstack(all_embeddings)