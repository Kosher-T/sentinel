"""
Smart Stacking: Data-Agnostic, Framework-Agnostic Input Grouping
================================================================
Shared utility for building input groups that respect folder structure
and model input requirements.

Used by:
- Golden Set Curator (for baseline generation)
- Drift Pipeline (for feature extraction)

Supports:
- Any data type (images, CSV, text, etc.)
- Any framework (Keras, PyTorch, ONNX)
- Any folder layout (flat, structured sequences, cobbled-together)

Author: Sentinel Project
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Union, Any

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] STACKING: %(message)s')

# --- Extension Families ---
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
TABULAR_EXTENSIONS = {'.csv', '.parquet', '.xlsx', '.tsv'}
TEXT_EXTENSIONS = {'.txt', '.json', '.xml', '.yaml', '.yml'}
ALL_DATA_EXTENSIONS = IMAGE_EXTENSIONS | TABULAR_EXTENSIONS | TEXT_EXTENSIONS


# =============================================================================
# MODEL INPUT SPEC DETECTION (Multi-Framework)
# =============================================================================

def get_model_input_specs(model_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Extract input specifications from a model file, regardless of framework.
    
    Detects framework from file extension and reads the expected input shape
    to determine stack_size (how many samples need to be concatenated for
    one forward pass).

    Args:
        model_path: Path to the model file (.keras, .h5, .pt, .pth, .onnx)

    Returns:
        Dict with keys: stack_size, target_h, target_w, expected_channels
        None if model cannot be loaded or specs cannot be determined
    """
    if model_path is None:
        return _default_specs()
    
    model_path = Path(model_path)
    if not model_path.exists():
        logging.warning(f"Model not found: {model_path}")
        return _default_specs()
    
    ext = model_path.suffix.lower()
    
    try:
        if ext in {'.keras', '.h5'}:
            return _specs_from_keras(model_path)
        elif ext in {'.pt', '.pth'}:
            return _specs_from_pytorch(model_path)
        elif ext in {'.onnx'}:
            return _specs_from_onnx(model_path)
        else:
            logging.warning(f"Unknown model format: {ext}. Using defaults.")
            return _default_specs()
    except Exception as e:
        logging.error(f"🔴 Failed to extract model specs: {e}")
        return _default_specs()


def _default_specs() -> Dict[str, Any]:
    """Return safe default specs (single image, no stacking)."""
    return {
        "stack_size": 1,
        "target_h": 224,
        "target_w": 224,
        "expected_channels": 3,
    }


def _compute_stack_size(channels: int) -> int:
    """Derive stack_size from channel count. 18ch = 6 images of 3ch each."""
    if channels > 3 and channels % 3 == 0:
        return channels // 3
    return 1


def _specs_from_keras(model_path: Path) -> Dict[str, Any]:
    """Extract specs from a Keras/TF model."""
    import keras  # type: ignore
    
    model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]  # Multi-input: use first
    
    # Keras shape: (batch, H, W, C)
    target_h = shape[1] if shape[1] else 224
    target_w = shape[2] if shape[2] else 224
    channels = shape[3] if shape[3] else 3
    
    specs = {
        "stack_size": _compute_stack_size(channels),
        "target_h": target_h,
        "target_w": target_w,
        "expected_channels": channels,
    }
    
    # Cleanup to free memory
    del model
    keras.backend.clear_session()
    
    logging.info(f"🟢 Keras specs: {specs['target_h']}×{specs['target_w']}, "
                 f"{specs['expected_channels']}ch, stack={specs['stack_size']}")
    return specs


def _specs_from_pytorch(model_path: Path) -> Dict[str, Any]:
    """Extract specs from a PyTorch model."""
    try:
        import torch  # type: ignore
    except ImportError:
        logging.warning("PyTorch not installed. Using default specs.")
        return _default_specs()
    
    model = torch.load(str(model_path), map_location='cpu')
    
    # Try to find the first Conv2d layer to infer input channels
    channels = 3  # Default
    if hasattr(model, 'modules'):
        for module in model.modules():
            if hasattr(module, 'in_channels'):
                channels = module.in_channels
                break
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dim() == 4:  # Conv layer: (out, in, kH, kW)
                    channels = module.weight.shape[1]
                    break
    
    specs = {
        "stack_size": _compute_stack_size(channels),
        "target_h": 224,   # PyTorch doesn't embed spatial dims in model
        "target_w": 224,
        "expected_channels": channels,
    }
    
    del model
    
    logging.info(f"🟢 PyTorch specs: {specs['expected_channels']}ch, stack={specs['stack_size']}")
    return specs


def _specs_from_onnx(model_path: Path) -> Dict[str, Any]:
    """Extract specs from an ONNX model."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        logging.warning("ONNX Runtime not installed. Using default specs.")
        return _default_specs()
    
    session = ort.InferenceSession(str(model_path))
    input_info = session.get_inputs()[0]
    shape = input_info.shape  # Typically (N, C, H, W) or (N, H, W, C)
    
    # Handle dynamic dimensions (may be strings like 'batch_size')
    dims = [d if isinstance(d, int) else None for d in shape]
    
    if len(dims) == 4:
        # Determine channel-first (NCHW) vs channel-last (NHWC)
        if dims[1] is not None and dims[1] <= 64:  # Likely channels
            channels = dims[1]
            target_h = dims[2] if dims[2] else 224
            target_w = dims[3] if dims[3] else 224
        else:
            channels = dims[3] if dims[3] else 3
            target_h = dims[1] if dims[1] else 224
            target_w = dims[2] if dims[2] else 224
    else:
        channels = 3
        target_h = 224
        target_w = 224
    
    specs = {
        "stack_size": _compute_stack_size(channels),
        "target_h": target_h,
        "target_w": target_w,
        "expected_channels": channels,
    }
    
    del session
    
    logging.info(f"🟢 ONNX specs: {specs['target_h']}×{specs['target_w']}, "
                 f"{specs['expected_channels']}ch, stack={specs['stack_size']}")
    return specs


# =============================================================================
# SMART GROUPING (Structure-Aware)
# =============================================================================

def _detect_extensions(directory: Path) -> set:
    """Auto-detect data extensions by scanning the first few files."""
    for f in directory.rglob('*'):
        if f.is_file() and not f.name.startswith('.'):
            ext = f.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                return IMAGE_EXTENSIONS
            elif ext in TABULAR_EXTENSIONS:
                return TABULAR_EXTENSIONS
            elif ext in TEXT_EXTENSIONS:
                return TEXT_EXTENSIONS
            else:
                # Unknown type — return just this extension
                return {ext}
    return set()


def _gather_files(directory: Path, extensions: set) -> List[str]:
    """Gather matching files directly in a directory (not recursive)."""
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            if f.suffix.lower() in extensions:
                files.append(str(f))
    return files


def build_groups(
    directory: Union[str, Path],
    stack_size: int,
    extensions: Optional[set] = None
) -> List:
    """
    Build input groups from a directory, respecting folder structure.

    Each subfolder is processed independently — its files are sorted and
    chunked into groups of stack_size. Root-level files are chunked the
    same way. Incomplete chunks (fewer than stack_size files) are dropped.

    This works correctly for:
    - Structured sequences (400 folders × 7 files → 400 groups of 6)
    - Cobbled-together batches (3 folders × 400/300/100 → 132 groups of 6)
    - Flat directories (2800 files → 466 groups of 6)

    Args:
        directory: Root directory to scan
        stack_size: Number of files per group (from model input requirements)
        extensions: File extensions to include. If None, auto-detect.

    Returns:
        List[str] if stack_size <= 1 (individual file paths)
        List[List[str]] if stack_size > 1 (grouped file paths)
    """
    directory = Path(directory)
    if not directory.exists():
        logging.error(f"🔴 Directory not found: {directory}")
        return []

    # Auto-detect extensions if not provided
    if extensions is None:
        extensions = _detect_extensions(directory)
        if not extensions:
            logging.warning(f"⚠️ No recognized files found in {directory}")
            return []
        logging.info(f"📂 Auto-detected extensions: {extensions}")

    # --- No stacking needed ---
    if stack_size <= 1:
        all_files = []
        # Root-level files
        all_files.extend(_gather_files(directory, extensions))
        # Subfolder files (recursive leaf files)
        for subfolder in sorted(directory.iterdir()):
            if subfolder.is_dir() and not subfolder.name.startswith('.'):
                for f in sorted(subfolder.rglob('*')):
                    if f.is_file() and f.suffix.lower() in extensions:
                        all_files.append(str(f))
        logging.info(f"📊 stack_size=1 → {len(all_files)} individual files")
        return all_files

    # --- Stacking mode ---
    groups = []
    total_files = 0
    total_dropped = 0

    # 1. Root-level files
    root_files = _gather_files(directory, extensions)
    if root_files:
        root_groups, dropped = _chunk_files(root_files, stack_size)
        groups.extend(root_groups)
        total_files += len(root_files)
        total_dropped += dropped

    # 2. Each subfolder independently
    subfolders = sorted([
        d for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    for subfolder in subfolders:
        # Gather files within this subfolder (recursive to handle nested structure)
        sub_files = []
        for f in sorted(subfolder.rglob('*')):
            if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in extensions:
                sub_files.append(str(f))

        if not sub_files:
            continue

        if len(sub_files) < stack_size:
            logging.debug(f"⏭️  Skipping {subfolder.name}: {len(sub_files)} files < stack_size {stack_size}")
            total_dropped += len(sub_files)
            continue

        sub_groups, dropped = _chunk_files(sub_files, stack_size)
        groups.extend(sub_groups)
        total_files += len(sub_files)
        total_dropped += dropped

    logging.info(
        f"📊 Built {len(groups)} groups of {stack_size} "
        f"from {total_files} files ({total_dropped} remainders dropped)"
    )
    return groups


def _chunk_files(files: List[str], chunk_size: int) -> tuple:
    """
    Split a sorted file list into complete chunks.
    
    Returns:
        (groups, dropped_count) — groups is List[List[str]], dropped is int
    """
    groups = []
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        if len(chunk) == chunk_size:
            groups.append(chunk)
    dropped = len(files) % chunk_size
    return groups, dropped
