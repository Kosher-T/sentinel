"""
Sentinel Setup
==============
Initial setup utility for configuring the Sentinel monitoring system.

This script handles first-time configuration:
- Create initial Golden Set from training data
- Validate all config paths exist
- Initialize required directories

Future expansions (not yet implemented):
- Cloud/local production environment hookup
- Deployment configuration
- Monitoring endpoints setup

Usage:
    python setup.py --model_path models/production/model.keras --training_data data/training

Author: Sentinel Project
"""

import sys
import argparse
import logging
from pathlib import Path

# --- Project Path Setup ---
file_path = Path(__file__).resolve()
project_root = file_path.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config
from services.golden_set_curator import GoldenSetCurator

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] SETUP: %(message)s'
)


def validate_directories() -> bool:
    """
    Validate and create required directories.
    
    Returns:
        True if all directories are valid/created
    """
    logging.info("📁 Validating directory structure...")
    
    directories = [
        ("BASE_DATA_DIR", config.BASE_DATA_DIR),
        ("GOLDEN_SET_DIR", config.GOLDEN_SET_DIR.parent),  # Parent, curator creates the dir
        ("ORIGINAL_DATA_PATH", config.ORIGINAL_DATA_PATH),
        ("INCOMING_DATA_PATH", config.INCOMING_DATA_PATH),
        ("ARCHIVED_DATA_PATH", config.ARCHIVED_DATA_PATH),
        ("MODEL_PATH", config.MODEL_PATH),
        ("OLD_MODEL_PATH", config.OLD_MODEL_PATH),
        ("FRESH_MODEL_PATH", config.FRESH_MODEL_PATH),
    ]
    
    all_valid = True
    for name, path in directories:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"   ✓ {name}: {path}")
        except Exception as e:
            logging.error(f"   ✗ {name}: {e}")
            all_valid = False
    
    return all_valid


def create_golden_set(model_path: Path, training_data: Path, sample_size: int) -> bool:
    """
    Create the initial Golden Set from training data.
    
    Args:
        model_path: Path to the production model
        training_data: Path to training data directory
        sample_size: Number of samples for the Golden Set
        
    Returns:
        True if Golden Set was created successfully
    """
    logging.info("🎯 Creating Initial Golden Set...")
    
    if not training_data.exists():
        logging.error(f"🔴 Training data not found: {training_data}")
        return False
    
    if not model_path.exists():
        logging.error(f"🔴 Model not found: {model_path}")
        return False
    
    try:
        curator = GoldenSetCurator(
            input_dirs=[training_data],
            model_path=model_path,
            sample_size=sample_size
        )
        exit_code = curator.curate()
        return exit_code == 0
    except Exception as e:
        logging.error(f"🔴 Golden Set creation failed: {e}")
        return False


def run_setup(args) -> int:
    """
    Run the full setup process.
    
    Returns:
        Exit code: 0 on success, 1 on failure
    """
    logging.info("=" * 60)
    logging.info("🚀 SENTINEL SETUP STARTING")
    logging.info("=" * 60)
    
    # Step 1: Validate directories
    if not validate_directories():
        logging.error("🔴 Directory validation failed")
        return 1
    
    # Step 2: Create Golden Set (if training data provided)
    if args.training_data:
        training_path = Path(args.training_data)
        model_path = Path(args.model_path) if args.model_path else None
        
        if model_path is None:
            # Try to find a model in production
            prod_models = list(config.OLD_MODEL_PATH.glob("*.keras")) + \
                         list(config.OLD_MODEL_PATH.glob("*.h5")) + \
                         list(config.OLD_MODEL_PATH.glob("*.pth"))
            if prod_models:
                model_path = prod_models[0]
                logging.info(f"📦 Using production model: {model_path.name}")
            else:
                logging.error("🔴 No model specified and none found in production folder")
                return 1
        
        if not create_golden_set(model_path, training_path, args.sample_size):
            logging.error("🔴 Golden Set creation failed")
            return 1
    else:
        logging.info("ℹ️ Skipping Golden Set creation (no --training_data provided)")
    
    logging.info("=" * 60)
    logging.info("✅ SENTINEL SETUP COMPLETE")
    logging.info("=" * 60)
    logging.info("")
    logging.info("Next steps:")
    logging.info("  1. Place production model in: models/production/")
    logging.info("  2. Start the distiller: python services/distiller.py")
    logging.info("  3. Run sentinel: python sentinel_watch.py")
    
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sentinel Setup - Initialize the monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup with Golden Set creation
  python setup.py --model_path models/production/model.keras --training_data data/training

  # Just validate directories (no Golden Set)
  python setup.py
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the production model for baseline generation"
    )
    
    parser.add_argument(
        "--training_data",
        type=str,
        default=None,
        help="Path to training data for initial Golden Set creation"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples for the Golden Set (default: 100)"
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_setup(args)


if __name__ == "__main__":
    sys.exit(main())
