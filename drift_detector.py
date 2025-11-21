import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import os

# --- CONFIGURATION ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32 # Adjust based on your local machine's RAM/VRAM

def create_embedding_model():
    """
    Creates a model based on MobileNetV2 that outputs a 1D embedding vector.
    
    This uses a pre-trained model (MobileNetV2) without its final
    classification layer. Instead, it uses GlobalAveragePooling2D
    to create a "feature vector" (embedding) for each image.
    """
    # 1. Load MobileNetV2, pre-trained on ImageNet
    #    include_top=False means we DONT want the final classification layer
    base_model = MobileNetV2(weights='imagenet', 
                           include_top=False, 
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # We don't need to re-train this model.
    base_model.trainable = False

    # 2. Add a new "top" to the model
    #    GlobalAveragePooling2D takes the [7, 7, 1280] tensor and makes it [1, 1280]
    #    This is how we get our "vector" for each image!
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = preprocess_input(inputs) # Apply MobileNet's specific pre-processing
    x = base_model(x, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x) # The key step!
    
    # 3. Create the new model
    embed_model = keras.Model(inputs, outputs)
    
    print("Embedding Model Created:")
    embed_model.summary()
    return embed_model

def get_image_paths(directory):
    """Recursively finds all image paths in a directory."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def generate_embeddings_from_directory(model, directory):
    """
    Generates embeddings for all images in a given directory.
    Uses batch processing to be memory-efficient.
    """
    print(f"\nGenerating embeddings for: {directory}")
    image_paths = get_image_paths(directory)
    
    if not image_paths:
        print(f"No images found in {directory}.")
        return None
        
    all_embeddings = []
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        
        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = image.img_to_array(img)
                batch_imgs.append(img_array)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
        
        if not batch_imgs:
            continue
            
        # Predict on the whole batch
        embeddings = model.predict(np.array(batch_imgs))
        all_embeddings.extend(embeddings)
        
        print(f"  Processed {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")

    return np.array(all_embeddings)

if __name__ == "__main__":
    # 1. Create the model
    embedding_model = create_embedding_model()
    
    # 2. Define your data directories (relative paths for local use)
    GOOD_DATA_DIR = 'data/extracted_frames'
    BAD_DATA_DIR = 'data/drifted_frames'
    
    # Check if directories exist
    if not os.path.exists(GOOD_DATA_DIR):
        print(f"Error: Good data directory not found at: {GOOD_DATA_DIR}")
        print("Please check your folder structure.")
    if not os.path.exists(BAD_DATA_DIR):
        print(f"Error: Bad data directory not found at: {BAD_DATA_DIR}")
        print("Please run data_saboteur.py or check your folder structure.")

    # 3. Generate the "Baseline" (Good Data)
    #    This is what we'll compare against in production
    if os.path.exists(GOOD_DATA_DIR):
        baseline_embeddings = generate_embeddings_from_directory(embedding_model, GOOD_DATA_DIR)
        
        if baseline_embeddings is not None and baseline_embeddings.size > 0:
            print(f"\nSuccessfully generated {baseline_embeddings.shape[0]} baseline embeddings.")
            # Save for later use!
            np.save('baseline_embeddings.npy', baseline_embeddings)
            print("Baseline embeddings saved to 'baseline_embeddings.npy'")
        else:
            print("No baseline embeddings were generated.")

    # 4. Generate "Drifted" embeddings (Bad Data)
    #    We do this now to prove our concept and find a threshold
    if os.path.exists(BAD_DATA_DIR):
        drifted_embeddings = generate_embeddings_from_directory(embedding_model, BAD_DATA_DIR)

        if drifted_embeddings is not None and drifted_embeddings.size > 0:
            print(f"\nSuccessfully generated {drifted_embeddings.shape[0]} drifted embeddings.")
            # Save for later use!
            np.save('drifted_embeddings.npy', drifted_embeddings)
            print("Drifted embeddings saved to 'drifted_embeddings.npy'")
        else:
            print("No drifted embeddings were generated.")
            
    print("\n--- Embedding generation complete ---")