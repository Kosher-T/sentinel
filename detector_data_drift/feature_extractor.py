import os
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
import all_config

# Factory for standard architectures
MODEL_FACTORY = {
    "MobileNetV2": {"class": MobileNetV2, "preprocess": mobile_preprocess},
    "VGG16": {"class": VGG16, "preprocess": vgg_preprocess},
    "ResNet50": {"class": ResNet50, "preprocess": resnet_preprocess}
}

def create_embedding_model(model_type=None, input_shape=None):
    """
    Initializes a pre-trained backbone. 
    Defaults to values in all_config if not explicitly provided.
    """
    # Fallback logic: Argument > Config File > Hardcoded Default
    m_type = model_type or (all_config.EMBEDDING_MODEL_TYPE if all_config else "MobileNetV2")
    i_shape = input_shape or (all_config.EMBEDDING_INPUT_SHAPE if all_config else (224, 224, 3))

    if m_type not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model: {m_type}. Options: {list(MODEL_FACTORY.keys())}")
    
    print(f"Feature Extractor: Loading {m_type} with ImageNet weights...")
    base = MODEL_FACTORY[m_type]["class"](
        weights='imagenet', 
        include_top=False, 
        input_shape=i_shape,
        pooling='avg'
    )
    return base

def get_image_paths(directory):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(valid_exts)
    ]

def generate_embeddings_from_directory(model, directory, batch_size=32, preprocess_mode=None):
    """
    Generates embeddings. If preprocess_mode is None, it tries to detect 
    the correct mode from all_config or defaults to MobileNetV2.
    """
    paths = get_image_paths(directory)
    if not paths:
        print(f"⚠️ No images found in {directory}")
        return np.array([])

    # Resolve preprocessing mode
    mode = preprocess_mode or (all_config.EMBEDDING_MODEL_TYPE if all_config else "MobileNetV2")
    
    if mode in MODEL_FACTORY:
        preprocess_func = MODEL_FACTORY[mode]["preprocess"]
    else:
        print(f"⚠️ Preprocess mode '{mode}' unknown. Using identity (no scaling).")
        preprocess_func = lambda x: x

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
            except Exception as e:
                continue

        if not batch_tensors:
            continue

        x = np.array(batch_tensors)
        x = preprocess_func(x)
        
        preds = model.predict(x, verbose=0)
        all_embeddings.append(preds)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])