import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
INPUT_DIR = 'data/extracted_frames'   # Folder with your clean, extracted frames
OUTPUT_DIR = 'data/drifted_frames'    # Where the "ruined" frames will go
DRIFT_TYPES = ['noise', 'blur', 'dark', 'low_res']

def add_gaussian_noise(frame):
    """Adds random grain/noise to the image."""
    row, col, ch = frame.shape
    mean = 0
    var = 50 # Adjust for more/less noise
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = frame + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_blur(frame):
    """Simulates out-of-focus or motion blur."""
    return cv2.GaussianBlur(frame, (15, 15), 0)

def make_darker(frame):
    """Simulates poor lighting conditions."""
    # Convert to HSV, lower the Value (brightness), convert back
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, -50) # Subtract 50 from brightness
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def lower_resolution(frame):
    """Simulates low-quality user uploads (downscale then upscale)."""
    height, width = frame.shape[:2]
    # Scale down to 25% size
    small = cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)
    # Scale back up (this introduces pixelation artifacts)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated

def process_image(input_path, output_path, drift_type):
    """Reads an image, applies a drift function, and saves the result."""
    try:
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            return

        # Apply the sabotage
        if drift_type == 'noise':
            frame = add_gaussian_noise(frame)
        elif drift_type == 'blur':
            frame = add_blur(frame)
        elif drift_type == 'dark':
            frame = make_darker(frame)
        elif drift_type == 'low_res':
            frame = lower_resolution(frame)
            
        cv2.imwrite(output_path, frame)
        
    except Exception as e:
        print(f"Error processing {input_path} for {drift_type}: {e}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        print("Please run frame_extractor.py first or check your directory names.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Iterate through each video's frame folder (e.g., 'data/extracted_frames/my_clip_1')
    for video_folder in os.listdir(INPUT_DIR):
        input_subfolder_path = os.path.join(INPUT_DIR, video_folder)
        
        if not os.path.isdir(input_subfolder_path):
            continue

        # Create a corresponding output subfolder (e.g., 'data/drifted_frames/my_clip_1')
        output_subfolder_path = os.path.join(OUTPUT_DIR, video_folder)
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)
            
        print(f"Processing frames in: {input_subfolder_path}")

        # Get all image files
        image_files = [f for f in os.listdir(input_subfolder_path) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print(f"No frames found in {input_subfolder_path}. Skipping.")
            continue

        for image_file in image_files:
            input_path = os.path.join(input_subfolder_path, image_file)
            base_filename = os.path.splitext(image_file)[0]
            extension = os.path.splitext(image_file)[1]
            
            # Create a drifted version for EACH type
            for drift in DRIFT_TYPES:
                output_filename = f"{drift}_{base_filename}{extension}"
                output_path = os.path.join(output_subfolder_path, output_filename)
                process_image(input_path, output_path, drift)

    print("Sabotage complete. Drifted frames generated.")

if __name__ == "__main__":
    main()