import cv2
import os
import argparse

# --- CONFIGURATION ---
DEFAULT_INPUT_DIR = './data/source_clips'
DEFAULT_OUTPUT_DIR = 'data/extracted_frames'
FRAME_SKIP_RATE = 1 # Set to 1 to extract every frame. Increase (e.g., 5) to only extract every 5th frame.

def extract_frames_from_video(video_path, output_folder, skip_rate):
    """Reads a video and saves selected frames to an output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0
    
    # Get the base name for saving frames
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Processing: {video_filename}")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % skip_rate == 0:
            frame_filename = os.path.join(output_folder, f"{video_filename}_frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Extraction complete for {video_filename}. Saved {saved_count} frames.")
    
def main():
    if not os.path.exists(DEFAULT_INPUT_DIR):
        os.makedirs(DEFAULT_INPUT_DIR)
        print(f"Created input directory: {DEFAULT_INPUT_DIR}. Please place your videos here.")
        return

    # Process all videos in the input directory
    video_files = [f for f in os.listdir(DEFAULT_INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]

    if not video_files:
        print(f"No videos found in {DEFAULT_INPUT_DIR}. Please add your source clips.")
        return

    # Ensure the main output directory exists
    if not os.path.exists(DEFAULT_OUTPUT_DIR):
        os.makedirs(DEFAULT_OUTPUT_DIR)
        
    for video_file in video_files:
        video_path = os.path.join(DEFAULT_INPUT_DIR, video_file)
        
        # Create a unique subfolder for each video's frames
        video_name = os.path.splitext(video_file)[0]
        output_subfolder = os.path.join(DEFAULT_OUTPUT_DIR, video_name)
        
        extract_frames_from_video(video_path, output_subfolder, FRAME_SKIP_RATE)

if __name__ == "__main__":
    main()