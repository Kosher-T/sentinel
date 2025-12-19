import os
import shutil
import glob
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data' / 'frames'
DEST_DIR = PROJECT_ROOT / 'data' / 'golden_set_inputs'
STRIDE = 50  # Skip 50 frames between samples to get variety (avoid nearly identical tests)
MAX_SAMPLES = 100  # Total number of golden samples to create (keeps the dataset light)

def setup_directories():
    """Creates the destination directory, wiping it if it exists to ensure a clean slate."""
    if os.path.exists(DEST_DIR):
        print(f"Cleaning existing directory: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    print(f"Created directory: {DEST_DIR}")

def get_sorted_frames(folder_path):
    """Returns a numerically sorted list of image paths from a folder."""
    # Extensions to look for
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Sort by filename to ensure temporal order (frame_001, frame_002...)
    return sorted(files)

def create_golden_pairs():
    setup_directories()
    
    # Get all clip folders (e.g., "1113(1)-1", "1113(10)-1")
    clip_folders = [f for f in glob.glob(os.path.join(SOURCE_DIR, '*')) if os.path.isdir(f)]
    
    print(f"Found {len(clip_folders)} source clips.")
    
    sample_count = 0
    
    for clip_path in clip_folders:
        if sample_count >= MAX_SAMPLES:
            break
            
        frames = get_sorted_frames(clip_path)
        
        # We need at least 3 frames to make a VFI triplet (1, 2, 3)
        # We use index i (im1) and i+2 (im3). The model generates i+1.
        
        # Iterate with stride to avoid redundant data
        for i in range(0, len(frames) - 2, STRIDE):
            if sample_count >= MAX_SAMPLES:
                break
                
            im1_path = frames[i]
            im3_path = frames[i+1] # VFI typically skips one frame to predict the middle, but this video is already
                                   # at 30 fps. No need to make interpolation results worse by sending half of the frames.  
            
            # Create the numbered folder (e.g., 0000001)
            folder_name = f"{sample_count + 1:07d}"
            target_folder = os.path.join(DEST_DIR, folder_name)
            os.makedirs(target_folder)
            
            # Copy and Rename
            shutil.copy2(im1_path, os.path.join(target_folder, "im1.jpg"))
            shutil.copy2(im3_path, os.path.join(target_folder, "im3.jpg"))
            
            # Create a metadata file so we know where this came from (for debugging)
            with open(os.path.join(target_folder, "source_info.txt"), "w") as f:
                f.write(f"Source Clip: {os.path.basename(clip_path)}\n")
                f.write(f"Original Im1: {os.path.basename(im1_path)}\n")
                f.write(f"Original Im3: {os.path.basename(im3_path)}\n")
            
            print(f"[{sample_count+1}/{MAX_SAMPLES}] Created pair in {folder_name} from {os.path.basename(clip_path)}")
            sample_count += 1

    print(f"\n--- Complete ---")
    print(f"Generated {sample_count} golden input pairs in '{DEST_DIR}'")
    print("Ready for Kaggle upload/Reference Generation.")

if __name__ == "__main__":
    create_golden_pairs()