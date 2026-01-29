import time
import sys
import argparse
import io
from pathlib import Path

# Force stdout to use UTF-8 to prevent UnicodeEncodeError on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def simulate_training(data_source, is_dir, output_model):
    source_type = "directory" if is_dir else "file"
    print(f"🟢 Training started using data {source_type}: {data_source}")
    
    # Simulate 5 epochs of training
    for epoch in range(1, 6):
        print(f"[TRAINER] Epoch {epoch}/5: Processing batch gradient...")
        time.sleep(1.2) # Simulate compute time
        
        # Simulate metrics output that Sentinel can log
        loss = 0.5 / epoch
        accuracy = 0.8 + (0.03 * epoch)
        print(f"[TRAINER] Epoch {epoch} Metrics: Loss={loss:.4f}, Accuracy={accuracy:.2f}")

    print("💾 Training complete. Saving model weights...")
    
    # Ensure the directory exists
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy model file
    with open(output_path, "w") as f:
        f.write("SENTINEL_DUMMY_WEIGHTS_v2.0")
    
    print(f"🟢 Model artifact secured at: {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated to match the arguments sent by execution_engine.py
    parser.add_argument("--data_file", type=str, help="Path to single training data file")
    parser.add_argument("--data_dir", type=str, help="Path to training data directory")
    parser.add_argument("--recursive", type=str, help="Recursive flag")
    
    args = parser.parse_args()

    # Determine which path was provided
    data_source = args.data_dir if args.data_dir else args.data_file
    is_dir = True if args.data_dir else False
    
    if not data_source:
        print("🔴 Error: No data source provided to trainer.")
        sys.exit(1)

    target_output = "models\\challenger\\challenger_v2.pth"
    
    try:
        simulate_training(data_source, is_dir, target_output)
        sys.exit(0) 
    except Exception as e:
        print(f"🔴 Training Interrupted: {e}")
        sys.exit(1)