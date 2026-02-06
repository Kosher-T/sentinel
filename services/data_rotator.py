"""
Data Rotator
============
Utility for rotating samples between datasets while maintaining a fixed size.

Used by:
- GoldenSetCurator for baseline updates
- SentinelWatch for ORIGINAL_DATA_PATH updates after successful deployments

The rotation strategy prevents datasets from growing infinitely while
ensuring they evolve to reflect new data distributions.

Author: Sentinel Project
"""

import sys
import shutil
import random
import logging
from pathlib import Path
from typing import List, Optional

# --- Project Path Setup ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] ROTATOR: %(message)s'
)


class DataRotator:
    """
    Handles rotation of samples between source and target directories.
    
    Rotation maintains a fixed-size target directory by replacing a percentage
    of existing samples with new ones from the source. This prevents:
    - Infinite growth of reference datasets
    - Stale baselines that don't reflect current data distributions
    
    Use Cases:
        1. Golden Set updates: Rotate production model outputs
        2. ORIGINAL_DATA_PATH updates: Add drifted data to prevent re-triggering
    """
    
    # Default rotation percentage
    DEFAULT_ROTATION_PERCENTAGE = 0.20
    
    def __init__(self, rotation_percentage: float = DEFAULT_ROTATION_PERCENTAGE):
        """
        Initialize the rotator.
        
        Args:
            rotation_percentage: Fraction of target samples to replace (0.0 to 1.0)
        """
        self.rotation_percentage = rotation_percentage
        self.stats = {
            "samples_added": 0,
            "samples_removed": 0,
            "samples_preserved": 0
        }
    
    def _collect_samples(self, directory: Path) -> List[Path]:
        """
        Collect all sample directories/files from a directory.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of sample paths
        """
        if not directory.exists():
            return []
        
        samples = []
        for item in directory.iterdir():
            if not item.name.startswith('.'):
                samples.append(item)
        return samples
    
    def _get_next_sample_number(self, target_dir: Path, prefix: str = "0_") -> int:
        """
        Find the next available sample number in target directory.
        
        Args:
            target_dir: Directory to check
            prefix: Sample naming prefix (e.g., "0_" or "sample_")
            
        Returns:
            Next available number
        """
        existing_nums = []
        for item in target_dir.iterdir():
            name = item.name
            if name.startswith(prefix):
                try:
                    num = int(name[len(prefix):])
                    existing_nums.append(num)
                except ValueError:
                    pass
        return max(existing_nums, default=0) + 1
    
    def rotate(
        self,
        source_dir: Path,
        target_dir: Path,
        sample_prefix: str = "0_"
    ) -> bool:
        """
        Rotate samples: replace percentage of target with samples from source.
        
        Args:
            source_dir: Directory containing new samples
            target_dir: Directory to update (reference dataset)
            sample_prefix: Naming prefix for new samples
            
        Returns:
            True if rotation succeeded
        """
        self.stats = {"samples_added": 0, "samples_removed": 0, "samples_preserved": 0}
        
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        if not source_dir.exists():
            logging.error(f"🔴 Source directory does not exist: {source_dir}")
            return False
        
        # Collect samples
        source_samples = self._collect_samples(source_dir)
        target_samples = self._collect_samples(target_dir)
        
        if not source_samples:
            logging.warning(f"⚠️ No samples in source directory: {source_dir}")
            return True  # Not an error, just nothing to do
        
        # Calculate rotation counts
        samples_to_remove = int(len(target_samples) * self.rotation_percentage)
        samples_to_add = min(samples_to_remove, len(source_samples))
        
        if samples_to_add == 0:
            samples_to_add = min(len(source_samples), max(1, int(len(target_samples) * 0.1)))
        
        logging.info(f"📊 Rotation: Remove {samples_to_remove}, Add {samples_to_add}")
        
        try:
            # Remove random samples from target
            if samples_to_remove > 0 and target_samples:
                to_delete = random.sample(target_samples, min(samples_to_remove, len(target_samples)))
                for sample in to_delete:
                    if sample.is_dir():
                        shutil.rmtree(sample)
                    else:
                        sample.unlink()
                    self.stats["samples_removed"] += 1
            
            self.stats["samples_preserved"] = len(target_samples) - self.stats["samples_removed"]
            
            # Add samples from source
            to_add = random.sample(source_samples, samples_to_add)
            next_num = self._get_next_sample_number(target_dir, sample_prefix)
            
            for sample in to_add:
                dest_name = f"{sample_prefix}{next_num:03d}"
                dest_path = target_dir / dest_name
                
                if sample.is_dir():
                    shutil.copytree(sample, dest_path)
                else:
                    shutil.copy2(sample, dest_path)
                
                self.stats["samples_added"] += 1
                next_num += 1
            
            logging.info(f"✅ Rotation complete: +{self.stats['samples_added']} "
                        f"-{self.stats['samples_removed']} "
                        f"={self.stats['samples_preserved']} preserved")
            return True
            
        except Exception as e:
            logging.error(f"🔴 Rotation failed: {e}")
            return False
    
    def add_samples(
        self,
        source_dir: Path,
        target_dir: Path,
        count: Optional[int] = None,
        sample_prefix: str = "0_"
    ) -> bool:
        """
        Add samples from source to target without removing any.
        
        Useful for ORIGINAL_DATA_PATH when you want to ensure drifted
        data is included in the reference baseline.
        
        Args:
            source_dir: Directory containing samples to add
            target_dir: Directory to add samples to
            count: Number of samples to add (None = all)
            sample_prefix: Naming prefix for new samples
            
        Returns:
            True if addition succeeded
        """
        self.stats = {"samples_added": 0, "samples_removed": 0, "samples_preserved": 0}
        
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        if not source_dir.exists():
            logging.error(f"🔴 Source directory does not exist: {source_dir}")
            return False
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        source_samples = self._collect_samples(source_dir)
        if not source_samples:
            logging.warning(f"⚠️ No samples in source: {source_dir}")
            return True
        
        # Select samples
        if count is not None and count < len(source_samples):
            to_add = random.sample(source_samples, count)
        else:
            to_add = source_samples
        
        try:
            next_num = self._get_next_sample_number(target_dir, sample_prefix)
            
            for sample in to_add:
                dest_name = f"{sample_prefix}{next_num:03d}"
                dest_path = target_dir / dest_name
                
                if sample.is_dir():
                    shutil.copytree(sample, dest_path)
                else:
                    shutil.copy2(sample, dest_path)
                
                self.stats["samples_added"] += 1
                next_num += 1
            
            logging.info(f"✅ Added {self.stats['samples_added']} samples to {target_dir.name}")
            return True
            
        except Exception as e:
            logging.error(f"🔴 Add samples failed: {e}")
            return False


def rotate_dataset(
    source: str,
    target: str,
    percentage: float = 0.20
) -> bool:
    """
    Convenience function for rotating datasets.
    
    Args:
        source: Path to source directory
        target: Path to target directory
        percentage: Rotation percentage (0.0 to 1.0)
        
    Returns:
        True if rotation succeeded
    """
    rotator = DataRotator(rotation_percentage=percentage)
    return rotator.rotate(Path(source), Path(target))
