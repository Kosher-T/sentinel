import tensorflow as tf
import keras
from keras.preprocessing import image
# Dynamic imports for supported architectures
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess
import numpy as np
import os
import model_config  # <--- Connects to your config file

# --- MODEL FACTORY ---
# This dictionary maps config names to actual Keras models and their specific preprocessors.
MODEL_FACTORY = {
    "MobileNetV2": {
        "class": MobileNetV2,
        "preprocess": mobile_preprocess
    },
    "VGG16": {
        "class": VGG16,
        "preprocess": vgg_preprocess
    },
    "ResNet50": {
        "class": ResNet50,
        "preprocess": resnet_preprocess
    }
}

BATCH_SIZE = 32

def create_embedding_model():
    """
    Creates a feature extraction model based on the configuration in model_config.py.
    
    It dynamically selects the architecture (MobileNet, VGG, etc.) and
    'bakes in' the correct preprocessing so the input remains raw RGB images.
    """
    model_type = model_config.EMBEDDING_MODEL_TYPE
    input_shape = model_config.EMBEDDING_INPUT_SHAPE
    
    # 1. Validate Config
    if model_type not in MODEL_FACTORY:
        raise ValueError(f"Model '{model_type}' is not supported in drift_detector.py. "
                         f"Available options: {list(MODEL_FACTORY.keys())}")
    
    print(f"--- Loading Feature Extractor: {model_type} ---")
    print(f"Input Shape: {input_shape}")
    
    selected_arch = MODEL_FACTORY[model_type]
    
    # 2. Load the Base Model (Pre-trained on ImageNet, No Classification Layer)
    base_model = selected_arch["class"](
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # We never want to retrain the feature extractor
    base_model.trainable = False

    # 3. Construct the Pipeline
    inputs = keras.Input(shape=input_shape)
    
    # Apply the specific preprocessing for this architecture (e.g., MobileNet vs VGG specific math)
    x = selected_arch["preprocess"](inputs)
    
    # Pass through the base model
    x = base_model(x, training=False)
    
    # 4. Vectorize (Flatten)
    # GlobalAveragePooling2D turns the 3D tensor output into a 1D vector
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    
    # 5. Final Model
    embed_model = keras.Model(inputs, outputs)
    embed_model.summary()
    
    return embed_model

def get_image_paths(directory):
    """Recursively finds all image paths in a directory."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def generate_embeddings_from_directory(model, directory):
    """
    Generates embeddings using the generic model provided.
    Reads input dimensions directly from the model config.
    """
    print(f"\nGenerating embeddings for: {directory}")
    image_paths = get_image_paths(directory)
    
    if not image_paths:
        print(f"No images found in {directory}.")
        return np.array([]) # Return empty array instead of None for safety
        
    all_embeddings = []
    
    # Read target size dynamically from config
    # input_shape is (224, 224, 3), we need just (224, 224) for load_img
    target_h, target_w = model_config.EMBEDDING_INPUT_SHAPE[:2]
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        
        for img_path in batch_paths:
            try:
                # Load image at the size required by the config
                img = image.load_img(img_path, target_size=(target_h, target_w))
                img_array = image.img_to_array(img)
                batch_imgs.append(img_array)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
        
        if not batch_imgs:
            continue
            
        # Predict on the whole batch
        # Note: We do NOT preprocess here explicitly anymore. 
        # The model created in create_embedding_model() handles it!
        embeddings = model.predict(np.array(batch_imgs), verbose=0)
        all_embeddings.extend(embeddings)
        
        print(f"  Processed {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")

    result = np.array(all_embeddings)
    
    # Validation check for feature count
    if result.size > 0 and result.shape[1] != model_config.EMBEDDING_FEATURE_COUNT:
         print(f"WARNING: Output feature count ({result.shape[1]}) does not match "
               f"config ({model_config.EMBEDDING_FEATURE_COUNT}). "
               f"Please update EMBEDDING_FEATURE_COUNT in model_config.py.")
               
    return result

if __name__ == "__main__":
    # Test run
    try:
        model = create_embedding_model()
        print("Model factory test successful.")
    except Exception as e:
        print(f"Model factory test failed: {e}")