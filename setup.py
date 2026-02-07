"""
Sentinel Setup Wizard
=====================
Interactive configuration wizard for the Sentinel monitoring system.

Run once at initial deployment to configure all_config.py:
- Calibrate drift threshold from training data
- Configure cloud platform and monitoring schedule
- Create initial Golden Set

Usage:
    python setup.py

After setup completes, run sentinel_watch.py to start monitoring.

Author: Sentinel Project
"""

import sys
import os
import re
import shutil
import tempfile
import random
import logging
from pathlib import Path
from collections import defaultdict

# --- Project Path Setup ---
file_path = Path(__file__).resolve()
project_root = file_path.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] SETUP: %(message)s'
)

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_SAMPLES = 200
GOLDEN_SET_SAMPLE_SIZE = 30
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
TABULAR_EXTENSIONS = {'.csv', '.parquet', '.xlsx'}

ALL_CONFIG_PATH = project_root / "all_config.py"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_banner():
    """Display welcome banner."""
    print("\n" + "=" * 60)
    print("🛡️  SENTINEL SETUP WIZARD")
    print("=" * 60)
    print("This wizard will configure Sentinel for your environment.")
    print("You'll need:")
    print("  • Training data (200+ samples)")
    print("  • Production model (.keras, .pth, or .onnx)")
    print("=" * 60 + "\n")


def prompt(message: str, default: str = None, validator=None) -> str:
    """
    Prompt user for input with optional default and validation.
    
    Args:
        message: Prompt message
        default: Default value (shown in brackets)
        validator: Function that returns (is_valid, error_message)
    
    Returns:
        User input or default
    """
    while True:
        if default:
            user_input = input(f"{message} [{default}]: ").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{message}: ").strip()
        
        if validator:
            is_valid, error_msg = validator(user_input)
            if not is_valid:
                print(f"  ❌ {error_msg}")
                continue
        
        return user_input


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """Prompt for yes/no with default."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{message} {suffix}: ").strip().lower()
    if not response:
        return default
    return response in ('y', 'yes')


def prompt_choice(message: str, options: list, default: int = 1) -> int:
    """Prompt for numbered choice."""
    print(f"\n{message}")
    for i, opt in enumerate(options, 1):
        marker = "→" if i == default else " "
        print(f"  {marker} [{i}] {opt}")
    
    while True:
        response = input(f"Choice [{default}]: ").strip()
        if not response:
            return default
        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        print(f"  ❌ Please enter a number between 1 and {len(options)}")


def validate_path_exists(path_str: str) -> tuple:
    """Validator: path must exist."""
    path = Path(path_str)
    if path.exists():
        return True, ""
    return False, f"Path not found: {path_str}"


def validate_model_path(path_str: str) -> tuple:
    """Validator: must be a valid model file."""
    path = Path(path_str)
    if not path.exists():
        return False, f"File not found: {path_str}"
    valid_exts = {'.keras', '.h5', '.pth', '.pt', '.onnx'}
    if path.suffix.lower() not in valid_exts:
        return False, f"Unsupported model format. Expected: {', '.join(valid_exts)}"
    return True, ""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(data_path: Path) -> list:
    """
    Load training data from various formats.
    
    Supports:
    - Directory of images (recursive)
    - CSV file (each row is a sample)
    - Directory with subdirectories (preserves structure for stratified split)
    
    Returns:
        List of file paths or list of (group_name, file_path) tuples
    """
    if data_path.is_file():
        # CSV or single file
        if data_path.suffix.lower() in TABULAR_EXTENSIONS:
            logging.info(f"📊 Loading tabular data from: {data_path.name}")
            # For tabular, we return the path and let drift pipeline handle it
            return [str(data_path)]
        else:
            return [str(data_path)]
    
    # Directory - gather files recursively with group info
    samples = []
    for file_path in data_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            # Group by parent directory for stratified sampling
            group = file_path.parent.name
            samples.append((group, str(file_path)))
    
    logging.info(f"📁 Found {len(samples)} samples across {len(set(g for g, _ in samples))} groups")
    return samples


def stratified_split(samples: list, ratio: float = 0.5) -> tuple:
    """
    Split samples while preserving group proportions.
    
    Args:
        samples: List of (group, path) tuples or list of paths
        ratio: Split ratio (default 0.5 for 50/50)
    
    Returns:
        (part_a, part_b) - two lists of file paths
    """
    # Handle simple list (no grouping)
    if samples and not isinstance(samples[0], tuple):
        random.shuffle(samples)
        split_idx = int(len(samples) * ratio)
        return samples[:split_idx], samples[split_idx:]
    
    # Group samples by their group name
    groups = defaultdict(list)
    for group, path in samples:
        groups[group].append(path)
    
    part_a = []
    part_b = []
    
    for group_name, paths in groups.items():
        random.shuffle(paths)
        split_idx = max(1, int(len(paths) * ratio))  # At least 1 in each part
        part_a.extend(paths[:split_idx])
        part_b.extend(paths[split_idx:])
    
    logging.info(f"📊 Stratified split: Part A = {len(part_a)}, Part B = {len(part_b)}")
    return part_a, part_b


# =============================================================================
# DRIFT CALIBRATION
# =============================================================================

def calibrate_drift_threshold(part_a: list, part_b: list, model_path: Path) -> float:
    """
    Calculate baseline drift by comparing two halves of training data.
    
    Returns:
        Recommended drift threshold
    """
    from detector_data_drift import pipeline as drift_pipeline
    
    # Create temporary directories for the two parts
    temp_dir = Path(tempfile.mkdtemp(prefix="sentinel_calibration_"))
    dir_a = temp_dir / "part_a"
    dir_b = temp_dir / "part_b"
    dir_a.mkdir()
    dir_b.mkdir()
    
    try:
        # Copy files to temp directories
        logging.info("📦 Preparing calibration data...")
        for i, path in enumerate(part_a):
            src = Path(path)
            dst = dir_a / f"{i:04d}{src.suffix}"
            shutil.copy2(src, dst)
        
        for i, path in enumerate(part_b):
            src = Path(path)
            dst = dir_b / f"{i:04d}{src.suffix}"
            shutil.copy2(src, dst)
        
        # Find distilled model if available
        latent_model = None
        distilled_dir = config.PRODUCTION_DISTILLED_DIR
        if distilled_dir.exists():
            latent_models = list(distilled_dir.glob("*_latent*"))
            if latent_models:
                latent_model = str(latent_models[0])
        
        # Run drift analysis
        logging.info("⚖️  Running drift calibration (Part A vs Part B)...")
        print("\n" + "-" * 40)
        
        drift_score, status = drift_pipeline.run_drift_analysis(
            baseline_path=dir_a,
            incoming_path=dir_b,
            force_recalc=True,
            latent_model_path=latent_model
        )
        
        print("-" * 40 + "\n")
        
        if drift_score is None or status == "ERROR":
            logging.error("🔴 Drift calibration failed")
            return None
        
        return drift_score
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def calculate_recommended_threshold(base_drift: float) -> float:
    """
    Calculate recommended threshold from base drift.
    
    Formula: base * 1.225 (10% tolerance + 12.5% buffer)
    """
    return round(base_drift * 1.225, 1)


# =============================================================================
# CONFIG WRITER
# =============================================================================

def update_all_config(settings: dict) -> bool:
    """
    Update all_config.py with the provided settings.
    
    Args:
        settings: Dict of variable_name -> value
    
    Returns:
        True if successful
    """
    try:
        with open(ALL_CONFIG_PATH, 'r') as f:
            content = f.read()
        
        for var_name, value in settings.items():
            # Format value appropriately
            if isinstance(value, str):
                formatted_value = f'"{value}"'
            elif isinstance(value, Path):
                formatted_value = f'PROJECT_ROOT / "{value.relative_to(project_root)}"'
            elif isinstance(value, list):
                formatted_value = repr(value)
            elif value is None:
                formatted_value = 'None'
            else:
                formatted_value = str(value)
            
            # Replace the variable assignment
            pattern = rf'^({var_name}\s*=\s*).*$'
            replacement = rf'\g<1>{formatted_value}'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Write back atomically
        temp_path = ALL_CONFIG_PATH.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            f.write(content)
        temp_path.replace(ALL_CONFIG_PATH)
        
        logging.info(f"✅ Updated {ALL_CONFIG_PATH.name}")
        return True
        
    except Exception as e:
        logging.error(f"🔴 Failed to update config: {e}")
        return False


# =============================================================================
# DIRECTORY VALIDATION
# =============================================================================

def validate_directories() -> bool:
    """Validate and create required directories."""
    logging.info("📁 Validating directory structure...")
    
    directories = [
        config.BASE_DATA_DIR,
        config.GOLDEN_SET_DIR.parent,
        config.ORIGINAL_DATA_PATH,
        config.INCOMING_DATA_PATH,
        config.ARCHIVED_DATA_PATH,
        config.MODEL_PATH,
        config.OLD_MODEL_PATH,
        config.FRESH_MODEL_PATH,
        config.MODEL_DECAY_ROOT,
        config.DRIFT_MONITOR_ROOT,
    ]
    
    for path in directories:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"❌ Failed to create {path}: {e}")
            return False
    
    logging.info("✅ All directories validated")
    return True


# =============================================================================
# GOLDEN SET CREATION
# =============================================================================

def create_golden_set(model_path: Path, training_data: list) -> bool:
    """Create initial Golden Set from training data samples."""
    from services.golden_set_curator import GoldenSetCurator
    
    # Get sample paths (handle both grouped and ungrouped)
    if training_data and isinstance(training_data[0], tuple):
        sample_paths = [path for _, path in training_data]
    else:
        sample_paths = training_data
    
    # Select samples for golden set
    samples = random.sample(sample_paths, min(GOLDEN_SET_SAMPLE_SIZE, len(sample_paths)))
    
    # Create temp directory with selected samples
    temp_dir = Path(tempfile.mkdtemp(prefix="golden_set_source_"))
    try:
        for i, path in enumerate(samples):
            src = Path(path)
            dst = temp_dir / f"{i:04d}{src.suffix}"
            shutil.copy2(src, dst)
        
        curator = GoldenSetCurator(
            input_dirs=[temp_dir],
            model_path=model_path,
            sample_size=GOLDEN_SET_SAMPLE_SIZE
        )
        exit_code = curator.curate()
        return exit_code == 0
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# MAIN SETUP FLOW
# =============================================================================

def run_setup() -> int:
    """Run the interactive setup wizard."""
    print_banner()
    
    # --- Step 1: Validate directories ---
    if not validate_directories():
        return 1
    
    # --- Step 2: Get training data path ---
    print("\n📂 TRAINING DATA")
    print("-" * 40)
    training_data_path = prompt(
        "Enter path to training data (folder or CSV)",
        validator=validate_path_exists
    )
    training_path = Path(training_data_path)
    
    # Load and validate training data
    samples = load_training_data(training_path)
    sample_count = len(samples)
    
    if sample_count < MIN_SAMPLES:
        print(f"\n❌ Insufficient training data: {sample_count} samples found")
        print(f"   Minimum required: {MIN_SAMPLES} samples")
        return 1
    
    print(f"✅ Loaded {sample_count} training samples")
    
    # --- Step 3: Get model path ---
    print("\n🧠 PRODUCTION MODEL")
    print("-" * 40)
    model_path_str = prompt(
        "Enter path to production model",
        validator=validate_model_path
    )
    model_path = Path(model_path_str)
    print(f"✅ Model: {model_path.name}")
    
    # --- Step 4: Drift calibration ---
    print("\n⚖️  DRIFT THRESHOLD CALIBRATION")
    print("-" * 40)
    print("Splitting training data and measuring baseline drift...")
    
    part_a, part_b = stratified_split(samples)
    base_drift = calibrate_drift_threshold(part_a, part_b, model_path)
    
    if base_drift is None:
        print("❌ Calibration failed. Using default threshold.")
        drift_threshold = 25.0
    else:
        recommended = calculate_recommended_threshold(base_drift)
        print(f"\n📊 Calibration Results:")
        print(f"   Base drift (within training data): {base_drift:.1f}%")
        print(f"   Recommended threshold: {recommended}%")
        
        if prompt_yes_no(f"Accept recommended threshold ({recommended}%)?", default=True):
            drift_threshold = recommended
        else:
            custom = prompt("Enter custom drift threshold", default=str(recommended))
            drift_threshold = float(custom)
    
    print(f"✅ Drift threshold: {drift_threshold}%")
    
    # --- Step 5: Decay threshold ---
    print("\n📉 DECAY THRESHOLD")
    print("-" * 40)
    decay_threshold = float(prompt(
        "Model decay threshold (%)",
        default="5.0"
    ))
    print(f"✅ Decay threshold: {decay_threshold}%")
    
    # --- Step 6: Execution platform ---
    print("\n☁️  EXECUTION PLATFORM")
    print("-" * 40)
    platform_choice = prompt_choice(
        "Select execution platform priority:",
        ["Local only", "Local → AWS", "Local → GCP", "AWS only", "GCP only"],
        default=1
    )
    
    platform_map = {
        1: ["LOCAL"],
        2: ["LOCAL", "AWS"],
        3: ["LOCAL", "GCP"],
        4: ["AWS"],
        5: ["GCP"],
    }
    execution_drivers = platform_map[platform_choice]
    print(f"✅ Execution priority: {' → '.join(execution_drivers)}")
    
    # --- Step 7: Monitoring schedule ---
    print("\n⏰ MONITORING SCHEDULE")
    print("-" * 40)
    schedule_choice = prompt_choice(
        "Select monitoring frequency:",
        ["Every 6 hours", "Every 12 hours", "Daily", "Custom cron"],
        default=1
    )
    
    schedule_map = {
        1: "0 */6 * * *",
        2: "0 */12 * * *",
        3: "0 0 * * *",
        4: None
    }
    
    if schedule_choice == 4:
        monitor_schedule = prompt("Enter cron expression", default="0 */6 * * *")
    else:
        monitor_schedule = schedule_map[schedule_choice]
    
    print(f"✅ Schedule: {monitor_schedule}")
    
    # --- Step 8: Golden Set ---
    print("\n🎯 GOLDEN SET")
    print("-" * 40)
    golden_set_path = prompt(
        "Path to existing Golden Set (leave blank to create from training)",
        default=""
    )
    
    if golden_set_path:
        if not Path(golden_set_path).exists():
            print(f"❌ Golden Set not found: {golden_set_path}")
            return 1
        print(f"✅ Using existing Golden Set: {golden_set_path}")
    else:
        print("Creating Golden Set from training data...")
        if create_golden_set(model_path, samples):
            print(f"✅ Golden Set created: {config.GOLDEN_SET_DIR}")
        else:
            print("❌ Failed to create Golden Set")
            return 1
    
    # --- Step 9: Write configuration ---
    print("\n💾 SAVING CONFIGURATION")
    print("-" * 40)
    
    settings = {
        'DRIFT_THRESHOLD': drift_threshold,
        'DECAY_THRESHOLD': decay_threshold,
        'EXECUTION_DRIVERS_PRIORITY': execution_drivers,
        'MONITOR_SCHEDULE': monitor_schedule,
        'RETRAIN_TRIGGER_COUNT': 5,
        'DRIFT_FAILURE_RATIO': 0.6,
        'TIMEFRAME_WINDOW': 5,
        'RETRAINING_SCRIPT': config.PROJECT_ROOT / "mock_train.py",
        'EXECUTION_TIMEOUT': 3600,
        'EXPECTED_CHALLENGER_PATH': config.FRESH_MODEL_PATH / "challenger_v2.pth",
    }
    
    if not update_all_config(settings):
        return 1
    
    # Copy model to production folder
    prod_model_dest = config.OLD_MODEL_PATH / model_path.name
    if not prod_model_dest.exists():
        shutil.copy2(model_path, prod_model_dest)
        print(f"✅ Model copied to: {prod_model_dest}")
    
    # --- Complete! ---
    print("\n" + "=" * 60)
    print("✅ SENTINEL SETUP COMPLETE!")
    print("=" * 60)
    print("\nConfiguration saved to: all_config.py")
    print("\nNext steps:")
    print("  1. Start the distiller: python services/distiller.py")
    print("  2. Run sentinel: python sentinel_watch.py")
    print("=" * 60 + "\n")
    
    return 0


def main() -> int:
    """Entry point."""
    try:
        return run_setup()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        return 1
    except Exception as e:
        logging.error(f"🔴 Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
