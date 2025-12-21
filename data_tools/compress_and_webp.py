import os
import cv2
import glob
from pathlib import Path

# --- CONFIGURATION ---
# The folder containing your septuplet sequences (001, 002, etc.)
GOLDEN_SET_DIR = input("Enter the path to the golden set directory (e.g., C:\\path\\to\\golden_set_septuplets\\sequences): ")

# Compression Settings
# WebP quality 80-85 is the sweet spot for VFI training data
TARGET_QUALITY = 85 

def optimize_septuplet_folder(folder_path, quality=85):
    """
    Optimizes all images in a single septuplet folder, 
    converting them to WebP to save space.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder_path, ext)))
    images = sorted(images)

    if not images:
        return 0, 0 # original_size, new_size

    orig_total = 0
    new_total = 0

    for i, img_path in enumerate(images):
        orig_total += os.path.getsize(img_path)
        
        img = cv2.imread(img_path)
        if img is None: continue

        # We keep the im1, im2 naming but switch to .webp
        save_name = f"im{i+1}.webp"
        save_path = os.path.join(folder_path, save_name)

        # Write optimized WebP
        cv2.imwrite(save_path, img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        
        # Remove the original file
        os.remove(img_path)
        new_total += os.path.getsize(save_path)
    
    return orig_total, new_total

def run_golden_set_optimization():
    print(f"🚀 Starting Golden Set Optimization")
    print(f"   Target: {GOLDEN_SET_DIR}")
    print(f"   Quality: {TARGET_QUALITY}")
    print("-" * 60)

    # Find all subfolders (001, 002, etc.)
    septuplet_folders = [f.path for f in os.scandir(GOLDEN_SET_DIR) if f.is_dir()]
    
    if not septuplet_folders:
        print("❌ No septuplet folders found. Check your path.")
        return

    print(f"🔍 Found {len(septuplet_folders)} sequences to process.")
    print(f"\n{'Folder':<10} | {'Original Size':<15} | {'New Size':<15} | {'Saved'}")
    print("-" * 60)

    grand_orig = 0
    grand_new = 0

    for folder in septuplet_folders:
        folder_name = os.path.basename(folder)
        orig_s, new_s = optimize_septuplet_folder(folder, TARGET_QUALITY)
        
        if orig_s == 0: continue
        
        grand_orig += orig_s
        grand_new += new_s
        
        saved_pct = (1 - (new_s / orig_s)) * 100
        print(f"{folder_name:<10} | {orig_s/1024:>7.0f} KB       | {new_s/1024:>7.0f} KB       | {saved_pct:.1f}%")

    print("-" * 60)
    total_saved_mb = (grand_orig - grand_new) / (1024 * 1024)
    total_pct = (1 - (grand_new / grand_orig)) * 100 if grand_orig > 0 else 0
    
    print(f"✨ COMPLETED")
    print(f"📊 Total Dataset Reduced from {grand_orig/(1024*1024):.2f} MB to {grand_new/(1024*1024):.2f} MB")
    print(f"📉 Total Space Recovered: {total_saved_mb:.2f} MB ({total_pct:.1f}%)")

if __name__ == "__main__":
    run_golden_set_optimization()