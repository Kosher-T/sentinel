import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from tensorflow.keras import layers

# --- Custom Model Components ---
# These definitions are copied directly from your training notebook
# to ensure the model loads correctly.

@tf.keras.utils.register_keras_serializable(package="Custom")
class SeparableKernelWarping(layers.Layer):
    """
    Custom layer for adaptive separable kernel warping in Video Frame Interpolation (VFI).

    This layer takes two input frames (I1, I3) and a set of predicted kernels,
    then applies separable convolution in two passes (horizontal and vertical)
    to synthesize the intermediate frame.
    """

    def __init__(self, kernel_size: int, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def call(self, inputs):
        """
        Args:
            inputs: tuple/list of (I1, I3, kernels)
                - I1: Tensor of shape (B, H, W, 3) → first frame
                - I3: Tensor of shape (B, H, W, 3) → second frame
                - kernels: Tensor of shape (B, H, W, 54) → predicted weights

        Returns:
            Tensor of shape (B, H, W, 3) → interpolated middle frame
        """
        # --- 1. Separate Inputs ---
        I1, I3, kernels = inputs
        B, H, W, C = tf.unstack(tf.shape(I1), num=4)
        K = self.kernel_size

        # --- 2. Reshape Kernels ---
        # Reshape to (B, H, W, 2 [I1/I3], K [weights], C [RGB])
        kernels_reshaped = tf.reshape(kernels, [B, H, W, 2, K, C])
        K1 = kernels_reshaped[..., 0, :, :]  # Kernels for I1
        K3 = kernels_reshaped[..., 1, :, :]  # Kernels for I3

        # --- 3. Horizontal Warping ---
        # Extract horizontal patches (1 × K window)
        ksizes_h = [1, 1, K, 1]
        strides_h = [1, 1, 1, 1]

        P1_H = tf.image.extract_patches(
            I1, sizes=ksizes_h, strides=strides_h, rates=[1, 1, 1, 1], padding="SAME"
        )
        P3_H = tf.image.extract_patches(
            I3, sizes=ksizes_h, strides=strides_h, rates=[1, 1, 1, 1], padding="SAME"
        )

        # Reshape patches to (B, H, W, K, C)
        P1_H = tf.reshape(P1_H, [B, H, W, K, C])
        P3_H = tf.reshape(P3_H, [B, H, W, K, C])

        # Apply horizontal kernels (dynamic convolution)
        I1_warped_H = tf.einsum("bhwkc,bhwkc->bhwc", P1_H, K1)
        I3_warped_H = tf.einsum("bhwkc,bhwkc->bhwc", P3_H, K3)

        # Combine intermediate results
        I_intermediate = I1_warped_H + I3_warped_H

        # --- 4. Vertical Warping ---
        # Extract vertical patches (K × 1 window)
        ksizes_v = [1, K, 1, 1]
        strides_v = [1, 1, 1, 1]

        P_V = tf.image.extract_patches(
            I_intermediate, sizes=ksizes_v, strides=strides_v, rates=[1, 1, 1, 1], padding="SAME"
        )
        P_V = tf.reshape(P_V, [B, H, W, K, C])

        # Simplified vertical kernel: average of K1 and K3
        K_V = (K1 + K3) / 2.0

        # Apply vertical kernels
        I_warped_V = tf.einsum("bhwkc,bhwkc->bhwc", P_V, K_V)

        return I_warped_V


    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config

# Get a small constant to prevent division by zero or log(0)
EPSILON = tf.keras.backend.epsilon()

@tf.keras.utils.register_keras_serializable(package="Custom")
def ssim_loss(y_true, y_pred):
    """
    Structural similarity loss.
    Returns 1 - SSIM so that higher similarity → lower loss.
    Includes clipping for numerical stability.
    """
    # Failsafe: Clip values to the expected [0, 1] range
    y_true_clipped = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    ssim_val = tf.image.ssim(y_true_clipped, y_pred_clipped, max_val=1.0) # shape (batch,)
    return 1.0 - ssim_val

@tf.keras.utils.register_keras_serializable(package="Custom")
def total_loss(y_true, y_pred):
    """
    The actual loss function that will be used by Keras/TensorFlow.
    Standardized logic (assuming default weights lambda_l1=0.8, lambda_ssim=0.2 
    as per your training script default).
    """
    lambda_l1 = 0.8
    lambda_ssim = 0.2

    # Failsafe: Clip values to the expected [0, 1] range
    y_true_clipped = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 1.0)

    # --- L1 Loss (Mean Absolute Error) ---
    l1_loss = tf.reduce_mean(tf.abs(y_true_clipped - y_pred_clipped))

    # --- SSIM Loss ---
    ssim_metric_per_image = tf.image.ssim(y_true_clipped, y_pred_clipped, max_val=1.0)
    structural_loss = tf.reduce_mean(1.0 - ssim_metric_per_image)

    # --- Weighted Combination ---
    composite_loss = (lambda_l1 * l1_loss) + (lambda_ssim * structural_loss)

    return composite_loss

@tf.keras.utils.register_keras_serializable(package="Custom")
def psnr_metric(y_true, y_pred):
    """
    A stable Peak Signal-to-Noise Ratio metric.
    Higher PSNR → better reconstruction quality.
    
    Handles potential `inf` values from perfect matches.
    """
    # Failsafe: Clip values to the expected [0, 1] range
    y_true_clipped = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    # Calculate PSNR
    psnr_val_per_image = tf.image.psnr(y_true_clipped, y_pred_clipped, max_val=1.0)
    
    # Handle potential `inf` values from perfect matches (where MSE=0)
    # Replace `inf` with a large, non-problematic number (e.g., 100.0)
    psnr_val_safe = tf.where(tf.math.is_inf(psnr_val_per_image), 100.0, psnr_val_per_image)
    
    return tf.reduce_mean(psnr_val_safe)


# --- Configuration ---
# Paths based on your Kaggle input structure
INPUT_ROOT = "/kaggle/input/golden-set-vfi-model/vfi_golden/golden_set_inputs"
MODEL_PATH = "/kaggle/input/golden-set-vfi-model/vfi_golden/vfi_epoch_99.keras"

# Output directory (Kaggle's writable directory is /kaggle/working)
OUTPUT_ROOT = "/kaggle/working/golden_set_complete"

# VFI Model Constraints
MODEL_INPUT_SIZE = (256, 256)

def load_vfi_model():
    """Loads the VFI model from the specified path, providing custom objects."""
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Define the dictionary of all custom objects needed for the model
    custom_objects = {
        'SeparableKernelWarping': SeparableKernelWarping,
        'total_loss': total_loss,
        'psnr_metric': psnr_metric,
        'ssim_loss': ssim_loss,
    }
    
    try:
        # 2. Pass the custom_objects dictionary to load_model
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_path, target_size):
    """
    Loads an image, resizes it to model input size, and normalizes it.
    Returns: Preprocessed image tensor (1, H, W, 3) and original dimensions.
    """
    # Load image in BGR (OpenCV default) and convert to RGB
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    
    # Resize to model's expected input size (256x256)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] and add batch dimension
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, (original_h, original_w)

def run_inference(model, im1_path, im3_path):
    """
    Runs the VFI model on a pair of images.
    Returns: The generated middle frame (numpy array, 0-255, RGB).
    """
    # Preprocess both inputs
    im1_batch, orig_dims = preprocess_image(im1_path, MODEL_INPUT_SIZE)
    im3_batch, _ = preprocess_image(im3_path, MODEL_INPUT_SIZE)
    
    # Stack inputs as expected by the model [batch, H, W, 6] or similar
    # Your model architecture usually takes concatenated inputs
    inputs = np.concatenate([im1_batch, im3_batch], axis=-1)
    
    # Run Inference
    prediction = model.predict(inputs, verbose=0)
    
    # Post-process output
    # Prediction is [1, 256, 256, 3], remove batch dim
    gen_img = prediction[0] 
    
    # Clip values to [0, 1] range to avoid artifacts
    gen_img = np.clip(gen_img, 0.0, 1.0)
    
    # Upscale back to original resolution
    # Note: We use cubic interpolation for better quality on upscale
    gen_img_upscaled = cv2.resize(gen_img, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_CUBIC)
    
    # Convert back to [0, 255] integers
    gen_img_uint8 = (gen_img_upscaled * 255.0).astype(np.uint8)
    
    return gen_img_uint8

def process_golden_set():
    """
    Main loop: Iterates through input folders, generates frames, and saves output.
    """
    # Create output directory
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    model = load_vfi_model()
    
    # Find all sample folders (e.g., 0000001, 0000002)
    sample_folders = sorted(glob(os.path.join(INPUT_ROOT, "*")))
    print(f"Found {len(sample_folders)} samples to process.")
    
    for folder in tqdm(sample_folders):
        folder_name = os.path.basename(folder)
        
        im1_path = os.path.join(folder, "im1.jpg")
        im3_path = os.path.join(folder, "im3.jpg")
        
        if not os.path.exists(im1_path) or not os.path.exists(im3_path):
            print(f"Skipping {folder_name}: Missing input images.")
            continue
            
        # Prepare output folder structure
        target_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            
        # Copy original inputs to the output folder (so we have a complete set)
        # This is important so your local monitor has everything in one place later
        shutil.copy2(im1_path, os.path.join(target_folder, "im1.jpg"))
        shutil.copy2(im3_path, os.path.join(target_folder, "im3.jpg"))
        
        # Generate the middle frame
        try:
            fresh_im2 = run_inference(model, im1_path, im3_path)
            
            # Save the generated result
            save_path = os.path.join(target_folder, "fresh_im2.jpg")
            # Convert RGB back to BGR for OpenCV saving
            cv2.imwrite(save_path, cv2.cvtColor(fresh_im2, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Failed to process sample {folder_name}: {e}")

    print("\n--- Processing Complete ---")
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("You can now zip and download the 'golden_set_complete' folder.")

if __name__ == "__main__":
    import shutil # Ensure shutil is imported
    process_golden_set()