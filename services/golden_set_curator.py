"""
Golden Set Curator
===================
A model-agnostic utility for managing the "Golden Set" — the ground-truth baseline
used to test model decay in Sentinel.

This script handles two states:
- CREATE: Initialize a new Golden Set from scratch
- UPDATE: Safely curate new samples into an existing set with backup + rotation

The Golden Set stores Input/Output pairs where the Output is the inference result
from the current Production Model. This allows the decay pipeline to compare 
challenger model outputs against a known-good baseline.

Usage:
    python golden_set_curator.py --input_dirs /path/to/data --model_path /path/to/model.keras
    
Author: Sentinel Project
"""

import sys
import os
import shutil
import zipfile
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# --- Project Path Setup ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] CURATOR: %(message)s'
)


class ModelInferenceWrapper:
    """
    Placeholder class for model inference.
    
    This wrapper abstracts the model inference process, allowing the Golden Set 
    Curator to work with ANY vision model architecture. The actual implementation
    should be adapted based on the specific model being monitored.
    
    The wrapper is responsible for:
    1. Loading the production model
    2. Running inference on input data
    3. Saving the output in a consistent format alongside the input
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize the inference wrapper.
        
        Args:
            model_path: Path to the production model file (.keras, .h5, .pt, etc.)
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the production model.
        
        NOTE: This is a placeholder. Implement actual model loading based on
        the framework being used (TensorFlow/Keras, PyTorch, ONNX, etc.)
        """
        logging.info(f"📦 Loading production model from: {self.model_path}")
        # Placeholder: Actual model loading would go here
        # Example for Keras:
        # from tensorflow.keras.models import load_model
        # self.model = load_model(self.model_path, compile=False)
        pass
    
    def generate_baseline(self, input_path: Path, output_dir: Path) -> bool:
        """
        Run inference on input data and save the result as baseline.
        
        This method takes an input sample (which could be a single image, a folder 
        of frames, or any other format), runs it through the production model,
        and saves the output alongside a copy of the input.
        
        Args:
            input_path: Path to the input sample (file or directory)
            output_dir: Directory where input + output pair should be saved
            
        Returns:
            True if baseline was generated successfully, False otherwise
        """
        try:
            logging.info(f"   -> Generating baseline for: {input_path.name}")
            
            # Create output directory structure
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for input and output
            input_dest = output_dir / "input"
            output_dest = output_dir / "output"
            input_dest.mkdir(exist_ok=True)
            output_dest.mkdir(exist_ok=True)
            
            # Copy input data
            if input_path.is_dir():
                # Copy entire directory contents
                for item in input_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, input_dest / item.name)
            else:
                # Copy single file
                shutil.copy2(input_path, input_dest / input_path.name)
            
            # --- PLACEHOLDER: Actual inference logic ---
            # Here you would:
            # 1. Load/preprocess the input
            # 2. Run self.model.predict() or equivalent
            # 3. Save the model output to output_dest
            #
            # Example pseudo-code:
            # input_data = preprocess(input_path)
            # prediction = self.model.predict(input_data)
            # save_prediction(prediction, output_dest / "prediction.npy")
            
            # For now, create a placeholder marker file
            marker_file = output_dest / ".baseline_generated"
            marker_file.write_text(f"Generated at: {datetime.now().isoformat()}\n"
                                   f"Model: {self.model_path.name}\n")
            
            return True
            
        except Exception as e:
            logging.error(f"🔴 Failed to generate baseline for {input_path.name}: {e}")
            return False


class GoldenSetCurator:
    """
    Manages Golden Set creation and safe updates.
    
    The Golden Set is a curated collection of Input/Output pairs that serve as
    the ground-truth baseline for model decay detection. This curator handles:
    
    - CREATE_MODE: Build a new Golden Set from input directories
    - UPDATE_MODE: Safely update existing set with rotation (replace 20% of samples)
    
    Rotation Strategy:
        To prevent the Golden Set from becoming stale or infinitely growing,
        UPDATE_MODE uses a rotation strategy: randomly replace 20% of existing
        samples with new samples from the input directories. This maintains a
        fixed size while allowing the baseline to evolve over time.
    
    Safety Features:
        - Timestamped .zip backup before any updates
        - Atomic writes via temp directory + rename
        - Path validation before processing
    """
    
    # Rotation percentage (how much of the old set to replace with new samples)
    ROTATION_PERCENTAGE = 0.20
    
    # Backup directory name
    BACKUPS_DIR_NAME = "backups"
    
    # Temp directory name for atomic writes
    TEMP_DIR_NAME = "temp_golden_set"
    
    def __init__(
        self,
        input_dirs: List[Path],
        model_path: Path,
        sample_size: int = 100,
        golden_set_dir: Optional[Path] = None
    ):
        """
        Initialize the curator.
        
        Args:
            input_dirs: List of directories containing new data to curate
            model_path: Path to the production model for baseline generation
            sample_size: Target number of samples in the Golden Set
            golden_set_dir: Override for Golden Set location (uses config default if None)
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.model_path = Path(model_path)
        self.sample_size = sample_size
        self.golden_set_dir = Path(golden_set_dir) if golden_set_dir else config.GOLDEN_SET_DIR
        self.backups_dir = config.BASE_DATA_DIR / self.BACKUPS_DIR_NAME
        self.temp_dir = config.BASE_DATA_DIR / self.TEMP_DIR_NAME
        
        self.mode = None  # Set during detect_mode()
        self.inference_wrapper = None  # Lazy-loaded
        
        # Statistics tracking
        self.stats = {
            "samples_added": 0,
            "samples_removed": 0,
            "samples_preserved": 0,
            "backup_created": None
        }
    
    def _validate_paths(self) -> bool:
        """
        Validate that all required paths exist.
        
        Returns:
            True if all paths are valid, False otherwise
        """
        # Check input directories
        for input_dir in self.input_dirs:
            if not input_dir.exists():
                logging.error(f"🔴 Input directory does not exist: {input_dir}")
                return False
            if not input_dir.is_dir():
                logging.error(f"🔴 Input path is not a directory: {input_dir}")
                return False
        
        # Check model path
        if not self.model_path.exists():
            logging.error(f"🔴 Model file does not exist: {self.model_path}")
            return False
        
        logging.info("✅ All paths validated successfully")
        return True
    
    def _detect_mode(self) -> str:
        """
        Detect whether to run in CREATE or UPDATE mode.
        
        Returns:
            "CREATE" if no Golden Set exists, "UPDATE" if it does
        """
        if not self.golden_set_dir.exists():
            self.mode = "CREATE"
            logging.info("📦 Mode: CREATE (no existing Golden Set found)")
        else:
            # Check if directory has any content
            contents = list(self.golden_set_dir.iterdir())
            if len(contents) == 0:
                self.mode = "CREATE"
                logging.info("📦 Mode: CREATE (Golden Set directory is empty)")
            else:
                self.mode = "UPDATE"
                logging.info(f"🔄 Mode: UPDATE (found {len(contents)} existing items)")
        
        return self.mode
    
    def _create_backup(self) -> Optional[Path]:
        """
        Create a timestamped .zip backup of the current Golden Set.
        
        Returns:
            Path to the backup file, or None if backup failed
        """
        if not self.golden_set_dir.exists():
            logging.warning("⚠️ No Golden Set to backup")
            return None
        
        try:
            # Ensure backup directory exists
            self.backups_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamped backup name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"golden_set_{timestamp}.zip"
            backup_path = self.backups_dir / backup_name
            
            logging.info(f"💾 Creating backup: {backup_name}")
            
            # Create zip archive
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.golden_set_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.golden_set_dir)
                        zipf.write(file_path, arcname)
            
            backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
            logging.info(f"✅ Backup created: {backup_name} ({backup_size_mb:.2f} MB)")
            self.stats["backup_created"] = str(backup_path)
            
            return backup_path
            
        except Exception as e:
            logging.error(f"🔴 Backup failed: {e}")
            return None
    
    def _collect_samples(self) -> List[Path]:
        """
        Collect all available samples from input directories.
        
        Samples can be either individual files or directories (for multi-file inputs
        like video frames or image sequences).
        
        Returns:
            List of paths to available samples
        """
        samples = []
        
        for input_dir in self.input_dirs:
            logging.info(f"🔍 Scanning: {input_dir}")
            
            for item in input_dir.iterdir():
                # Skip hidden files and system files
                if item.name.startswith('.'):
                    continue
                
                # Include both files and directories as valid samples
                samples.append(item)
        
        logging.info(f"📊 Found {len(samples)} potential samples in input directories")
        return samples
    
    def _sample_random(self, items: List[Path], count: int) -> List[Path]:
        """
        Randomly sample items from a list.
        
        Args:
            items: List of items to sample from
            count: Number of items to sample
            
        Returns:
            Randomly selected subset of items
        """
        if len(items) <= count:
            return items
        return random.sample(items, count)
    
    def _get_existing_samples(self) -> List[Path]:
        """
        Get list of existing samples in the Golden Set.
        
        Returns:
            List of paths to existing samples
        """
        if not self.golden_set_dir.exists():
            return []
        
        samples = []
        for item in self.golden_set_dir.iterdir():
            if not item.name.startswith('.'):
                samples.append(item)
        
        return samples
    
    def _curate_create_mode(self) -> bool:
        """
        Create a new Golden Set from scratch.
        
        Returns:
            True if creation succeeded, False otherwise
        """
        logging.info("🚀 Starting CREATE mode...")
        
        # Collect all available samples
        all_samples = self._collect_samples()
        if not all_samples:
            logging.error("🔴 No samples found in input directories")
            return False
        
        # Random selection
        selected = self._sample_random(all_samples, self.sample_size)
        logging.info(f"📋 Selected {len(selected)} samples for Golden Set")
        
        # Initialize inference wrapper
        self.inference_wrapper = ModelInferenceWrapper(self.model_path)
        
        # Use atomic writes: build in temp dir first
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)
        
        try:
            # Process each selected sample
            success_count = 0
            for i, sample_path in enumerate(selected, 1):
                sample_dest = self.temp_dir / f"sample_{i:04d}"
                
                if self.inference_wrapper.generate_baseline(sample_path, sample_dest):
                    success_count += 1
                    self.stats["samples_added"] += 1
                
                if i % 10 == 0:
                    logging.info(f"   Progress: {i}/{len(selected)} samples processed")
            
            if success_count == 0:
                logging.error("🔴 No samples were processed successfully")
                shutil.rmtree(self.temp_dir)
                return False
            
            # Atomic rename: temp -> final
            if self.golden_set_dir.exists():
                shutil.rmtree(self.golden_set_dir)
            self.temp_dir.rename(self.golden_set_dir)
            
            logging.info(f"✅ Golden Set created with {success_count} samples")
            return True
            
        except Exception as e:
            logging.error(f"🔴 CREATE mode failed: {e}")
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            return False
    
    def _curate_update_mode(self) -> bool:
        """
        Update an existing Golden Set with rotation strategy.
        
        Rotation: Replace ROTATION_PERCENTAGE (20%) of existing samples with new ones.
        This maintains a fixed-size set while allowing the baseline to evolve.
        
        Returns:
            True if update succeeded, False otherwise
        """
        logging.info("🔄 Starting UPDATE mode...")
        
        # CRITICAL: Create backup first
        backup_path = self._create_backup()
        if backup_path is None:
            logging.error("🔴 Backup failed - aborting update for safety")
            return False
        
        # Collect new samples
        new_samples = self._collect_samples()
        if not new_samples:
            logging.warning("⚠️ No new samples found - Golden Set unchanged")
            return True
        
        # Get existing samples
        existing_samples = self._get_existing_samples()
        existing_count = len(existing_samples)
        
        # Calculate rotation counts
        samples_to_remove = int(existing_count * self.ROTATION_PERCENTAGE)
        samples_to_add = min(samples_to_remove, len(new_samples))
        
        logging.info(f"📊 Rotation plan: Remove {samples_to_remove}, Add {samples_to_add}")
        
        if samples_to_add == 0:
            logging.info("ℹ️ No rotation needed (not enough new samples)")
            return True
        
        # Initialize inference wrapper
        self.inference_wrapper = ModelInferenceWrapper(self.model_path)
        
        # Use atomic writes: copy to temp, modify, then swap
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        try:
            # Copy current Golden Set to temp
            shutil.copytree(self.golden_set_dir, self.temp_dir)
            
            # Select samples to remove (random)
            temp_samples = list(self.temp_dir.iterdir())
            temp_samples = [s for s in temp_samples if not s.name.startswith('.')]
            samples_to_delete = self._sample_random(temp_samples, samples_to_remove)
            
            # Remove selected old samples
            for sample in samples_to_delete:
                if sample.is_dir():
                    shutil.rmtree(sample)
                else:
                    sample.unlink()
                self.stats["samples_removed"] += 1
            
            self.stats["samples_preserved"] = existing_count - len(samples_to_delete)
            
            # Add new samples
            new_selected = self._sample_random(new_samples, samples_to_add)
            
            # Find next available sample number
            existing_nums = []
            for item in self.temp_dir.iterdir():
                if item.name.startswith("sample_"):
                    try:
                        num = int(item.name.split("_")[1])
                        existing_nums.append(num)
                    except (ValueError, IndexError):
                        pass
            next_num = max(existing_nums, default=0) + 1
            
            for sample_path in new_selected:
                sample_dest = self.temp_dir / f"sample_{next_num:04d}"
                next_num += 1
                
                if self.inference_wrapper.generate_baseline(sample_path, sample_dest):
                    self.stats["samples_added"] += 1
            
            # Atomic swap: remove old, rename temp to final
            shutil.rmtree(self.golden_set_dir)
            self.temp_dir.rename(self.golden_set_dir)
            
            logging.info(f"✅ Update complete: +{self.stats['samples_added']} "
                        f"-{self.stats['samples_removed']} "
                        f"={self.stats['samples_preserved']} preserved")
            return True
            
        except Exception as e:
            logging.error(f"🔴 UPDATE mode failed: {e}")
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            logging.info("💡 Original Golden Set preserved (backup available at: "
                        f"{backup_path})")
            return False
    
    def curate(self) -> int:
        """
        Main entry point: run the curation process.
        
        Returns:
            Exit code: 0 on success, 1 on failure
        """
        logging.info("=" * 60)
        logging.info("🎯 GOLDEN SET CURATOR STARTING")
        logging.info("=" * 60)
        
        try:
            # Step 1: Validate paths
            if not self._validate_paths():
                return 1
            
            # Step 2: Detect mode
            self._detect_mode()
            
            # Step 3: Run appropriate mode
            if self.mode == "CREATE":
                success = self._curate_create_mode()
            else:
                success = self._curate_update_mode()
            
            # Step 4: Report results
            logging.info("=" * 60)
            if success:
                logging.info("✅ CURATION COMPLETE")
                logging.info(f"   Mode: {self.mode}")
                logging.info(f"   Samples Added: {self.stats['samples_added']}")
                if self.mode == "UPDATE":
                    logging.info(f"   Samples Removed: {self.stats['samples_removed']}")
                    logging.info(f"   Samples Preserved: {self.stats['samples_preserved']}")
                    if self.stats['backup_created']:
                        logging.info(f"   Backup: {self.stats['backup_created']}")
                logging.info("=" * 60)
                return 0
            else:
                logging.error("🔴 CURATION FAILED")
                logging.info("=" * 60)
                return 1
                
        except Exception as e:
            logging.error(f"🔴 Unexpected error: {e}")
            return 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Golden Set Curator - Manage ground-truth baselines for model decay detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new Golden Set from history data
  python golden_set_curator.py --input_dirs data/data_drift/history --model_path models/production/model.keras

  # Update existing set with new samples (100 target size)
  python golden_set_curator.py --input_dirs data/incoming --model_path models/production/model.keras --sample_size 100

  # Multiple input directories
  python golden_set_curator.py --input_dirs data/history data/new_batch --model_path models/production/model.keras
        """
    )
    
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing data to curate into the Golden Set"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the production model used to generate baseline predictions"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Target number of samples in the Golden Set (default: 100)"
    )
    
    parser.add_argument(
        "--golden_set_dir",
        type=str,
        default=None,
        help="Override Golden Set directory (uses config.GOLDEN_SET_DIR by default)"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    curator = GoldenSetCurator(
        input_dirs=args.input_dirs,
        model_path=args.model_path,
        sample_size=args.sample_size,
        golden_set_dir=args.golden_set_dir
    )
    
    return curator.curate()


if __name__ == "__main__":
    sys.exit(main())
