# Uses a config file to set up a Keras feature extraction model.
# Supports multiple architectures (MobileNet, VGG, ResNet) via a factory pattern.
# Uses the loaded model to generate embeddings from images in a directory.

import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess
import numpy as np
import os
from pathlib import Path
import sys

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
import all_config

MODEL_FACTORY = {
    "MobileNetV2": {"class": MobileNetV2,
                    "preprocess": mobile_preprocess},
    "VGG16": {"class": VGG16,
              "preprocess": vgg_preprocess},
    "ResNet50": {"class": ResNet50,
                 "preprocess": resnet_preprocess}
}

BATCH_SIZE = 32

def create_embedding_model():
    """Creates a feature extraction model based on model_config.py."""
    cfg_type = all_config.EMBEDDING_MODEL_TYPE
    if cfg_type not in MODEL_FACTORY:
        raise ValueError(f"Model {cfg_type} not supported.")
    
    base = MODEL_FACTORY[cfg_type]["class"](
        weights='imagenet', 
        include_top=False, 
        input_shape=all_config.EMBEDDING_INPUT_SHAPE,
        pooling='avg'
    )
    return base

def extract_features(model, image_paths):
    """Processes a list of image paths and returns a numpy array of embeddings."""
    all_embeddings = []
    target_h, target_w = all_config.EMBEDDING_INPUT_SHAPE[:2]
    
    # Get the specific preprocessor for the current model type
    preprocess_func = MODEL_FACTORY[all_config.EMBEDDING_MODEL_TYPE]["preprocess"]

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_imgs = []
        
        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=(target_h, target_w))
                img_array = image.img_to_array(img)
                batch_imgs.append(img_array)
            except Exception as e:
                print(f"⚠️ Could not load {img_path}: {e}")
        
        if not batch_imgs: continue
            
        # Standardize and Predict
        x = np.array(batch_imgs)
        x = preprocess_func(x)
        embeddings = model.predict(x, verbose=0)
        all_embeddings.extend(embeddings)
        
    return np.array(all_embeddings)