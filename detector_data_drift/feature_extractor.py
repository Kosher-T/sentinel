import os
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess

# Factory for standard architectures used in Drift Detection
MODEL_FACTORY = {
    "MobileNetV2": {"class": MobileNetV2,
                    "preprocess": mobile_preprocess},
    "VGG16": {"class": VGG16,
              "preprocess": vgg_preprocess},
    "ResNet50": {"class": ResNet50,
                 "preprocess": resnet_preprocess}
}

def create_embedding_model(model_type="MobileNetV2", input_shape=(224, 224, 3)):
    """
    DRIFT PIPELINE: Use ImageNet weights to see general visual changes.
    DECAY PIPELINE: Can also be used to see if the 'style' of data has changed.
    """
    if model_type not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model: {model_type}")
    
    print(f"Sentinel: Initializing {model_type} backbone...")
    base = MODEL_FACTORY[model_type]["class"](
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape,
        pooling='avg'
    )
    return base

def get_image_paths(directory):
    """Unified recursive discovery for both pipelines."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                paths.append(os.path.join(root, file))
    return sorted(paths)

def generate_embeddings(model, directory, preprocess_mode="MobileNetV2", batch_size=32):
    """
    Core engine for both Data Drift and Model Decay.
    - For Drift: use generic ImageNet preprocessing.
    - For Decay: if using your own model, ensure preprocess_mode matches your training logic.
    """
    paths = get_image_paths(directory)
    if not paths:
        return np.array([])

    # Determine preprocessing logic
    if preprocess_mode in MODEL_FACTORY:
        preprocess_func = MODEL_FACTORY[preprocess_mode]["preprocess"]
    else:
        # Fallback to simple scaling if custom model is used
        preprocess_func = lambda x: x / 255.0

    target_size = model.input_shape[1:3]
    all_embeddings = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_tensors = []
        
        for p in batch_paths:
            try:
                img = image.load_img(p, target_size=target_size)
                img_array = image.img_to_array(img)
                batch_tensors.append(img_array)
            except Exception:
                continue # Skip corrupt files silently for automation

        if not batch_tensors: continue

        x = np.array(batch_tensors)
        x = preprocess_func(x)
        
        # This works for Keras Applications OR your custom loaded .h5/.keras models
        preds = model.predict(x, verbose=0)
        all_embeddings.append(preds)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])

if __name__ == "__main__":
    pass