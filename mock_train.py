import time
import sys
import argparse
import io
import os
from pathlib import Path

# Force stdout to use UTF-8 to prevent UnicodeEncodeError on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def simulate_training(data_source, is_dir, output_model):
    source_type = "directory" if is_dir else "file"
    print(f"🟢 Training started using data {source_type}: {data_source}")

    # --- DANGEROUS STRATEGY BLOCK (For Sentinel Testing) ---
    try:
        import tensorflow as tf
        print("⚠️ Initializing Distributed Training Context...")
        # This is the 'dangerous' keyword Sentinel scans for
        strategy = tf.distribute.MirroredStrategy() 
        print(f"⚠️ MirroredStrategy active. Number of devices: {strategy.num_replicas_in_sync}")
    except Exception as e:
        print(f"⚠️ Note: MirroredStrategy initialization skipped or failed ({e}). Proceeding with CPU/Single-GPU.")

    # Simulate 5 epochs of training
    for epoch in range(1, 6):
        print(f"[TRAINER] Epoch {epoch}/5: Processing batch gradient...")
        time.sleep(1.5) # Simulate compute time
        
        # Simulated metrics
        loss = 0.5 / epoch
        accuracy = 0.8 + (0.01 * epoch)

        # TRIGGER: Simulated Numerical Instability for testing Fail-Fast logic
        # If testing 'nan' detection, we flip the loss at epoch 4
        if epoch == 4:
            loss = float('nan')
            print(f"[TRAINER] Epoch {epoch} Metrics: Loss={loss}, Accuracy={accuracy:.2f}")
            print("🔴 CRITICAL: Numerical instability detected in weights. Terminating...")
            sys.exit(1)

        print(f"[TRAINER] Epoch {epoch} Metrics: Loss={loss:.4f}, Accuracy={accuracy:.2f}")

    print("💾 Training complete. Saving model weights...")
    
    # Ensure the directory exists
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy model file
    with open(output_path, "w") as f:
        f.write("SENTINEL_DUMMY_WEIGHTS_v2.0_DISTRIBUTED")
    
    print(f"🟢 Model artifact secured at: {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="Path to single training data file")
    parser.add_argument("--data_dir", type=str, help="Path to training data directory")
    parser.add_argument("--recursive", type=str, help="Recursive flag")
    
    args = parser.parse_args()

    # Determine which path was provided
    data_source = args.data_dir if args.data_dir else args.data_file
    is_dir = True if args.data_dir else False
    
    # Default output path if not provided by env
    output_model = os.getenv("CHALLENGER_PATH", "models/challenger/challenger_v2.pth")

    if not data_source:
        print("🔴 Error: No data source provided to mock_train.py")
        sys.exit(1)

    simulate_training(data_source, is_dir, output_model)