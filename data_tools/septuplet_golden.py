import os
import shutil
import glob
from pathlib import Path

# --- Configuration ---
# Based on your local paths provided
SOURCE_DIR = r"C:\Code\Code\Python\frame_generation_engine\sentinel\data\frames"
DEST_DIR = r"C:\Code\Code\Python\frame_generation_engine\sentinel\data\golden_set_septuplets"

STRIDE = 60        # Skip ~2 seconds of 30fps video between samples for variety
MAX_SAMPLES = 50   # Number of 7-frame sequences to create
SEQUENCE_LEN = 7   # We need exactly 7 frames for the Septuplet model

def setup_directories():
    """Wipes the destination to ensure the golden set is fresh and organized."""
    if os.path.exists(DEST_DIR):
        print(f"🧹 Cleaning old golden set directory...")
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    print(f"📂 Created: {DEST_DIR}")

def get_sorted_frames(folder_path):
    """Numerically sorts frames to maintain temporal order."""
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    # Sorts based on filename (e.g., frame_001.jpg, frame_002.jpg)
    return sorted(files)

def create_septuplet_set():
    setup_directories()
    
    # Find all subdirectories in the frames folder (the individual clips)
    clip_folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
    
    sample_count = 0
    
    for clip_path in clip_folders:
        if sample_count >= MAX_SAMPLES:
            break
            
        frames = get_sorted_frames(clip_path)
        
        # We need at least 7 frames to make one sequence
        if len(frames) < SEQUENCE_LEN:
            continue
            
        # Iterate through frames with a stride
        for i in range(0, len(frames) - SEQUENCE_LEN, STRIDE):
            if sample_count >= MAX_SAMPLES:
                break
                
            # Create folder name with 001, 002 naming scheme
            folder_name = f"{sample_count + 1:03d}"
            target_folder = os.path.join(DEST_DIR, folder_name)
            os.makedirs(target_folder)
            
            # Copy 7 consecutive frames and rename them to im1.jpg ... im7.jpg
            for j in range(SEQUENCE_LEN):
                src_frame = frames[i + j]
                dest_frame = os.path.join(target_folder, f"im{j+1}.jpg")
                shutil.copy2(src_frame, dest_frame)
            
            # Optional: Add a breadcrumb file for tracking
            with open(os.path.join(target_folder, "info.txt"), "w") as f:
                f.write(f"Source: {os.path.basename(clip_path)}\n")
                f.write(f"Original Start Frame: {os.path.basename(frames[i])}")
                
            sample_count += 1
            print(f"✅ Generated Sequence {folder_name} from {os.path.basename(clip_path)}")

    print(f"\n✨ Done! {sample_count} septuplet sequences ready in {DEST_DIR}")

if __name__ == "__main__":
    create_septuplet_set()