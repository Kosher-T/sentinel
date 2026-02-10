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
from detector_data_drift.smart_stacking import build_groups, get_model_input_specs

# --- Framework Imports (handled gracefully) ---
try:
    import numpy as np
except ImportError:
    np = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
    # Enable unsafe deserialization for custom layers
    try:
        tf.keras.config.enable_unsafe_deserialization()  # type: ignore
    except Exception:
        pass
    KERAS_AVAILABLE = True
except ImportError:
    tf = None
    KERAS_AVAILABLE = False

try:
    import torch  # type: ignore
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] CURATOR: %(message)s'
)


class ModelInferenceWrapper:
    """
    Multi-framework inference wrapper for vision models.
    
    Supports automatic detection and loading of:
    - Keras/TensorFlow models (.keras, .h5)
    - PyTorch models (.pt, .pth)
    - ONNX models (.onnx)
    
    The wrapper abstracts inference across frameworks, allowing the Golden Set 
    Curator to work with ANY vision model architecture.
    
    Framework Detection:
        The model framework is automatically detected based on file extension:
        - .keras, .h5 → TensorFlow/Keras
        - .pt, .pth → PyTorch
        - .onnx → ONNX Runtime
    """
    
    # Supported extensions by framework
    KERAS_EXTENSIONS = {'.keras', '.h5'}
    PYTORCH_EXTENSIONS = {'.pt', '.pth'}
    ONNX_EXTENSIONS = {'.onnx'}
    
    # Common image extensions for preprocessing
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, model_path: Path):
        """
        Initialize the inference wrapper and load the model.
        
        Args:
            model_path: Path to the production model file
            
        Raises:
            ValueError: If the model format is not supported
            RuntimeError: If the required framework is not installed
        """
        self.model_path = Path(model_path)
        self.model = None
        self.framework = None
        self.input_shape = None
        
        # Get input specs (stack_size, dimensions) from model
        specs = get_model_input_specs(model_path)
        self.stack_size = specs["stack_size"] if specs else 1
        self.target_h = specs["target_h"] if specs else 224
        self.target_w = specs["target_w"] if specs else 224
        self.expected_channels = specs["expected_channels"] if specs else 3
        
        self._detect_framework()
        self._load_model()
        
        logging.info(f"   Stack size: {self.stack_size}, Target: {self.target_h}×{self.target_w}, Channels: {self.expected_channels}")
    
    def _detect_framework(self) -> str:
        """
        Detect the model framework based on file extension.
        
        Returns:
            Framework name: 'keras', 'pytorch', or 'onnx'
            
        Raises:
            ValueError: If extension is not recognized
        """
        ext = self.model_path.suffix.lower()
        
        if ext in self.KERAS_EXTENSIONS:
            if not KERAS_AVAILABLE:
                raise RuntimeError(
                    f"Keras/TensorFlow is required for {ext} models but not installed. "
                    "Install with: pip install tensorflow"
                )
            self.framework = 'keras'
        elif ext in self.PYTORCH_EXTENSIONS:
            if not PYTORCH_AVAILABLE:
                raise RuntimeError(
                    f"PyTorch is required for {ext} models but not installed. "
                    "Install with: pip install torch"
                )
            self.framework = 'pytorch'
        elif ext in self.ONNX_EXTENSIONS:
            if not ONNX_AVAILABLE:
                raise RuntimeError(
                    f"ONNX Runtime is required for {ext} models but not installed. "
                    "Install with: pip install onnxruntime"
                )
            self.framework = 'onnx'
        else:
            raise ValueError(
                f"Unsupported model format: {ext}. "
                f"Supported: {self.KERAS_EXTENSIONS | self.PYTORCH_EXTENSIONS | self.ONNX_EXTENSIONS}"
            )
        
        logging.info(f"🔍 Detected framework: {self.framework.upper()}")
        return self.framework
    
    def _load_model(self):
        """
        Load the model using the appropriate framework.
        """
        logging.info(f"📦 Loading production model from: {self.model_path}")
        
        if self.framework == 'keras':
            self._load_keras_model()
        elif self.framework == 'pytorch':
            self._load_pytorch_model()
        elif self.framework == 'onnx':
            self._load_onnx_model()
    
    def _load_keras_model(self):
        """Load a Keras/TensorFlow model."""
        try:
            self.model = keras_load_model(self.model_path, compile=False, safe_mode=False)
            # Extract input shape for preprocessing
            try:
                self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
                logging.info(f"   Input shape: {self.input_shape}")
            except Exception:
                self.input_shape = None
            logging.info(f"✅ Keras model loaded successfully")
        except Exception as e:
            logging.error(f"🔴 Failed to load Keras model: {e}")
            raise
    
    def _load_pytorch_model(self):
        """Load a PyTorch model."""
        try:
            # Try loading as a full model first, then as state_dict
            self.model = torch.load(self.model_path, map_location='cpu')
            if hasattr(self.model, 'eval'):
                self.model.eval()
            logging.info(f"✅ PyTorch model loaded successfully")
        except Exception as e:
            logging.error(f"🔴 Failed to load PyTorch model: {e}")
            raise
    
    def _load_onnx_model(self):
        """Load an ONNX model using ONNX Runtime."""
        try:
            self.model = ort.InferenceSession(str(self.model_path))
            # Get input shape from ONNX model
            input_info = self.model.get_inputs()[0]
            self.input_shape = input_info.shape[1:]  # Remove batch dimension
            logging.info(f"   Input name: {input_info.name}, shape: {self.input_shape}")
            logging.info(f"✅ ONNX model loaded successfully")
        except Exception as e:
            logging.error(f"🔴 Failed to load ONNX model: {e}")
            raise
    
    def _preprocess_input(self, input_paths) -> Optional[Any]:
        """
        Preprocess input data for inference.
        
        Handles:
        - Single file path (str or Path) → load as single image
        - List of paths (group) → load and stack on channel axis for multi-input models
        - Directory path → load all images in directory
        
        Args:
            input_paths: Single path, list of paths, or directory path
            
        Returns:
            Preprocessed numpy array ready for inference, shape (1, H, W, C)
        """
        if np is None:
            logging.error("NumPy is required for preprocessing")
            return None
        
        try:
            # Normalize input to a list of file paths
            file_paths = self._resolve_input_paths(input_paths)
            
            if not file_paths:
                logging.warning(f"⚠️ No valid files resolved from input")
                return None
            
            # Load all images
            images = []
            for fp in file_paths:
                img = self._load_image(Path(fp))
                if img is not None:
                    images.append(img)
            
            if not images:
                logging.warning(f"⚠️ No images loaded successfully")
                return None
            
            # Build the input tensor
            if self.stack_size > 1 and len(images) >= self.stack_size:
                # Channel-axis concatenation: (H, W, 3) × N → (H, W, 3N)
                stacked = np.concatenate(images[:self.stack_size], axis=-1)
                batch = np.expand_dims(stacked, axis=0)  # (1, H, W, C*N)
            else:
                # Single image or stack_size=1: standard batch
                batch = np.stack(images, axis=0)  # (N, H, W, C)
            
            # Normalize to [0, 1] if not already
            if batch.max() > 1.0:
                batch = batch.astype(np.float32) / 255.0
            
            return batch
            
        except Exception as e:
            logging.error(f"🔴 Preprocessing failed: {e}")
            return None
    
    def _resolve_input_paths(self, input_paths) -> List[str]:
        """
        Normalize various input types to a flat list of file paths.
        
        Accepts:
        - str or Path (single file or directory)
        - List[str] or List[Path] (group of files)
        """
        if isinstance(input_paths, (str, Path)):
            p = Path(input_paths)
            if p.is_dir():
                return sorted([
                    str(f) for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
                ])
            elif p.is_file():
                return [str(p)]
            return []
        
        if isinstance(input_paths, list):
            return [str(p) for p in input_paths if Path(p).is_file()]
        
        return []
    
    def _load_image(self, image_path: Path) -> Optional[Any]:
        """
        Load and resize a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Numpy array of shape (H, W, C)
        """
        try:
            if Image is not None:
                # Use PIL for loading
                img = Image.open(image_path).convert('RGB')
                
                # Resize if we know the expected input shape
                if self.input_shape is not None:
                    # input_shape is typically (H, W, C) for Keras or (C, H, W) for PyTorch
                    if len(self.input_shape) >= 2:
                        h, w = self.input_shape[0], self.input_shape[1]
                        if h is not None and w is not None and h > 0 and w > 0:
                            img = img.resize((w, h))
                
                return np.array(img, dtype=np.float32)
            else:
                # Fallback: try cv2
                import cv2  # type: ignore
                img = cv2.imread(str(image_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img.astype(np.float32)
                
        except Exception as e:
            logging.warning(f"⚠️ Could not load image {image_path.name}: {e}")
            return None
    
    def _run_inference(self, preprocessed_input: Any) -> Optional[Any]:
        """
        Run inference using the loaded model.
        
        Args:
            preprocessed_input: Preprocessed numpy array
            
        Returns:
            Model output (numpy array)
        """
        try:
            if self.framework == 'keras':
                return self._infer_keras(preprocessed_input)
            elif self.framework == 'pytorch':
                return self._infer_pytorch(preprocessed_input)
            elif self.framework == 'onnx':
                return self._infer_onnx(preprocessed_input)
        except Exception as e:
            logging.error(f"🔴 Inference failed: {e}")
            return None
    
    def _infer_keras(self, inputs: Any) -> Any:
        """Run inference with Keras model."""
        return self.model.predict(inputs, verbose=0)
    
    def _infer_pytorch(self, inputs: Any) -> Any:
        """Run inference with PyTorch model."""
        # Convert to tensor and adjust dimensions if needed (NHWC -> NCHW)
        tensor_input = torch.from_numpy(inputs)
        if tensor_input.dim() == 4 and tensor_input.shape[-1] in [1, 3, 4]:
            # Likely NHWC format, convert to NCHW
            tensor_input = tensor_input.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            output = self.model(tensor_input)
        
        # Convert back to numpy
        if hasattr(output, 'numpy'):
            return output.numpy()
        elif hasattr(output, 'cpu'):
            return output.cpu().numpy()
        return output
    
    def _infer_onnx(self, inputs: Any) -> Any:
        """Run inference with ONNX Runtime."""
        input_name = self.model.get_inputs()[0].name
        
        # ONNX might expect NCHW format
        if inputs.shape[-1] in [1, 3, 4] and len(inputs.shape) == 4:
            inputs = np.transpose(inputs, (0, 3, 1, 2))
        
        outputs = self.model.run(None, {input_name: inputs.astype(np.float32)})
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _save_output(self, output: Any, output_dir: Path) -> bool:
        """
        Save model output to disk.
        
        Saves as:
        - .npy file for numeric arrays
        - .png if output appears to be an image
        
        Args:
            output: Model output to save
            output_dir: Directory to save output in
            
        Returns:
            True if saved successfully
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Always save raw output as .npy
            npy_path = output_dir / "prediction.npy"
            np.save(npy_path, output)
            
            # If output looks like an image, save as PNG too
            if len(output.shape) >= 3:
                try:
                    # Take first output if batched
                    img_data = output[0] if output.shape[0] <= 8 else output
                    
                    # Handle NCHW format
                    if img_data.shape[0] in [1, 3, 4] and len(img_data.shape) == 3:
                        img_data = np.transpose(img_data, (1, 2, 0))
                    
                    # Normalize to [0, 255]
                    if img_data.max() <= 1.0:
                        img_data = (img_data * 255).astype(np.uint8)
                    else:
                        img_data = np.clip(img_data, 0, 255).astype(np.uint8)
                    
                    # Save as image
                    if Image is not None:
                        if img_data.shape[-1] == 1:
                            img_data = img_data.squeeze(-1)
                        pil_img = Image.fromarray(img_data)
                        pil_img.save(output_dir / "prediction.png")
                except Exception:
                    pass  # Image saving is optional
            
            return True
            
        except Exception as e:
            logging.error(f"🔴 Failed to save output: {e}")
            return False
    
    def generate_baseline(self, input_data, output_dir: Path) -> bool:
        """
        Run inference on input data and save the result as baseline.
        
        This method:
        1. Copies the input to output_dir/input/
        2. Preprocesses the input for the model
        3. Runs inference
        4. Saves the model output to output_dir/output/
        
        Args:
            input_data: Path to input sample (file or dir), or list of paths (group)
            output_dir: Directory where input + output pair should be saved
            
        Returns:
            True if baseline was generated successfully, False otherwise
        """
        try:
            # Determine label for logging
            if isinstance(input_data, list):
                label = f"group of {len(input_data)} files"
            else:
                label = Path(input_data).name
            logging.info(f"   -> Generating baseline for: {label}")
            
            # Create output directory structure
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for input and output
            input_dest = output_dir / "input"
            output_dest = output_dir / "output"
            input_dest.mkdir(exist_ok=True)
            output_dest.mkdir(exist_ok=True)
            
            # Copy input data
            if isinstance(input_data, list):
                # Group of files
                for p in input_data:
                    src = Path(p)
                    if src.is_file():
                        shutil.copy2(src, input_dest / src.name)
            elif Path(input_data).is_dir():
                for item in Path(input_data).iterdir():
                    if item.is_file():
                        shutil.copy2(item, input_dest / item.name)
            else:
                src = Path(input_data)
                shutil.copy2(src, input_dest / src.name)
            
            # Preprocess input
            preprocessed = self._preprocess_input(input_data)
            if preprocessed is None:
                logging.warning(f"⚠️ Could not preprocess {label}, saving input only")
                marker = output_dest / ".preprocessing_failed"
                marker.write_text(f"Preprocessing failed at: {datetime.now().isoformat()}\n")
                return True
            
            # Run inference
            output = self._run_inference(preprocessed)
            if output is None:
                logging.warning(f"⚠️ Inference failed for {label}, saving input only")
                marker = output_dest / ".inference_failed"
                marker.write_text(f"Inference failed at: {datetime.now().isoformat()}\n")
                return True
            
            # Save output
            if not self._save_output(output, output_dest):
                logging.warning(f"⚠️ Could not save output for {label}")
                return True  # Input was saved
            
            # Success marker
            marker = output_dest / ".baseline_generated"
            marker.write_text(
                f"Generated at: {datetime.now().isoformat()}\n"
                f"Model: {self.model_path.name}\n"
                f"Framework: {self.framework}\n"
                f"Stack size: {self.stack_size}\n"
                f"Input shape: {preprocessed.shape}\n"
                f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}\n"
            )
            
            return True
            
        except Exception as e:
            logging.error(f"🔴 Failed to generate baseline for {label}: {e}")
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
    
    def _collect_samples(self) -> List:
        """
        Collect and group samples from input directories using smart stacking.
        
        Uses build_groups() to respect folder structure and model stack_size.
        Returns groups when stack_size > 1, individual paths when stack_size == 1.
        
        Returns:
            List of groups (List[List[str]]) or individual paths (List[str])
        """
        all_groups = []
        stack_size = 1
        
        # Get stack_size from inference wrapper if available
        if self.inference_wrapper is not None:
            stack_size = self.inference_wrapper.stack_size
        
        for input_dir in self.input_dirs:
            logging.info(f"🔍 Scanning: {input_dir}")
            groups = build_groups(input_dir, stack_size)
            all_groups.extend(groups)
        
        logging.info(f"📊 Found {len(all_groups)} sample groups (stack_size={stack_size})")
        return all_groups
    
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
        
        # Initialize inference wrapper FIRST (needed for stack_size in _collect_samples)
        self.inference_wrapper = ModelInferenceWrapper(self.model_path)
        
        # Collect all available samples (uses smart stacking with model's stack_size)
        all_samples = self._collect_samples()
        if not all_samples:
            logging.error("🔴 No samples found in input directories")
            return False
        
        # Random selection
        selected = self._sample_random(all_samples, self.sample_size)
        logging.info(f"📋 Selected {len(selected)} samples for Golden Set")
        
        # Use atomic writes: build in temp dir first
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)
        
        try:
            # Process each selected sample (may be a group or individual path)
            success_count = 0
            for i, sample_data in enumerate(selected, 1):
                sample_dest = self.temp_dir / f"sample_{i:04d}"
                
                if self.inference_wrapper.generate_baseline(sample_data, sample_dest):
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
        
        # Initialize inference wrapper (needed for stack_size)
        self.inference_wrapper = ModelInferenceWrapper(self.model_path)
        
        # Re-collect samples with correct stack_size now that wrapper is loaded
        new_samples = self._collect_samples()
        
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
