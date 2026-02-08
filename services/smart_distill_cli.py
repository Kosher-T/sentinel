import os
import sys
import json
import hashlib
import numpy as np
import logging
from pathlib import Path
import tensorflow as tf

# Setup Environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import all_config as config

try:
    from tensorflow.keras.models import Model, load_model # type:ignore
except ImportError:
    from keras.models import Model, load_model

# Bypass Security
try:
    if hasattr(tf, 'keras'):
        tf.keras.config.enable_unsafe_deserialization() # type: ignore
    elif 'keras' in globals():
        import keras
        keras.config.enable_unsafe_deserialization()
except Exception:
    pass

MEMORY_FILE = file_path.parent / "distiller_memory.json"

def get_fingerprint(model):
    """Creates a unique hash based on the layer architecture sequence."""
    sig = []
    for l in model.layers:
        try:
            # Safely handle InputLayer or layers with missing metadata
            shape_str = str(getattr(l, 'output_shape', 'dynamic'))
        except Exception:
            shape_str = "unknown"
        sig.append(f"{l.__class__.__name__}:{shape_str}")
    
    full_sig = "|".join(sig)
    return hashlib.md5(full_sig.encode()).hexdigest()

def load_memory():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(mem):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(mem, f, indent=4)
    print(f"💾 Knowledge saved to {MEMORY_FILE.name}")

def analyze_model_layers(model):
    """
    Returns a ranked list of candidate layers for truncation.
    Ranked from Best (ID 0) to Worst based on scoring heuristics.
    """
    candidates = []
    layers = model.layers
    
    print(f"\nAnalyzing {len(layers)} layers for cut points...")
    
    for i, layer in enumerate(layers):
        if i < 1 or i == len(layers) - 1:
            continue
            
        score = 0
        name_lower = layer.name.lower()
        type_name = layer.__class__.__name__
        
        try:
            out_shape = getattr(layer, 'output_shape', None)
        except Exception:
            out_shape = None

        # --- Scoring Heuristics (Preference Order) ---
        
        # 1. Higher score for layers following a BatchNormalization (Standard Head pattern)
        if i > 0 and 'batch' in layers[i-1].name.lower():
            score += 15
        
        # 2. Prefer Bottlenecks / Pooling / Flatten
        if 'pooling' in name_lower or 'flatten' in name_lower:
            score += 10
        
        # 3. Prefer Activation layers (Clean feature maps)
        if 'activation' in name_lower or 'relu' in name_lower:
            score += 8
            
        # 4. Shape-based preference (Vectors > Tensors)
        if out_shape and len(out_shape) == 2:
            score += 8
        elif out_shape and len(out_shape) == 4:
            score += 2
            
        candidates.append({
            'index': i,
            'name': layer.name,
            'type': type_name,
            'shape': str(out_shape) if out_shape else "Dynamic",
            'score': score,
            'layer_obj': layer
        })

    # Primary Sort: Score (Descending)
    # Secondary Sort: Index (Descending) - favors deeper layers on tie
    candidates.sort(key=lambda x: (x['score'], x['index']), reverse=True)
    return candidates

def test_truncation(model, layer_name):
    """Runs the variance check on a temporary cut with dynamic input shape detection."""
    try:
        target_layer = model.get_layer(layer_name)
        cut_model = Model(inputs=model.input, outputs=target_layer.output)
        
        # Detect required input shape directly from the model
        # This handles 18-channel VFI models or non-standard inputs automatically
        try:
            input_shape = model.input_shape
            if isinstance(input_shape, list): 
                input_shape = input_shape[0]
            
            # Convert list/tuple to actual values, replacing None with 224
            processed_shape = []
            for dim in input_shape[1:]: # Skip batch dimension
                processed_shape.append(dim if dim is not None else 224)
            
            test_input = np.random.rand(5, *processed_shape).astype(np.float32)
        except Exception:
            # Hard fallback
            test_input = np.random.rand(5, 224, 224, 3).astype(np.float32)

        features = cut_model.predict(test_input, verbose=0) # type: ignore
        
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
            
        variance = np.var(features, axis=0).mean()
        return True, variance, features.shape
    except Exception as e:
        return False, str(e), None

def run_interactive_session():
    print("\n" + "="*60)
    print("🧠 SENTINEL SMART DISTILLER TRAINER")
    print("="*60)
    print("Note: Candidates are ranked from Best (ID 0) to Worst based on logic.")
    
    models_found = []
    for path in config.DISTILL_MAP.keys():
        p = Path(path)
        if p.exists():
            models_found.extend(list(p.glob("*.keras")))
            
    if not models_found:
        print("🔴 No models found in configured source folders.")
        return

    print(f"Found {len(models_found)} models.")
    memory = load_memory()
    
    for model_path in models_found:
        print(f"\n🔹 Processing: {model_path.name}")
        
        try:
            model = load_model(str(model_path), compile=False, safe_mode=False)
            fp = get_fingerprint(model)
            
            if fp in memory:
                known_layer = memory[fp]
                print(f"   🟢 Known Architecture! Preference: '{known_layer}'")
                choice = input(f"   Do you want to re-teach this model? (y/N): ")
                if choice.lower() != 'y':
                    continue

            candidates = analyze_model_layers(model)
            
            print(f"\n   {'ID':<4} | {'Score':<5} | {'Layer Name':<25} | {'Type':<15} | {'Output Shape'}")
            print("   " + "-"*80)
            
            top_n = candidates[:10]
            for idx, c in enumerate(top_n):
                print(f"   {idx:<4} | {c['score']:<5} | {c['name']:<25} | {c['type']:<15} | {c['shape']}")
            
            print("\n   Options: [0-9] to pick (ranked best to worst), [s] to skip, [q] to quit")
            selection = input("   Select cut point: ")
            
            if selection.lower() == 'q': break
            if selection.lower() == 's': continue
            
            try:
                sel_idx = int(selection)
                selected_cand = top_n[sel_idx]
                layer_name = selected_cand['name']
                
                print(f"   🧪 Testing cut at '{layer_name}'...")
                success, var_or_err, shape = test_truncation(model, layer_name)
                
                if success:
                    print(f"   🟢 Test Passed! Variance: {var_or_err:.6f} | Shape: {shape}")
                    confirm = input("   Save this architecture preference? (Y/n): ")
                    if confirm.lower() != 'n':
                        memory[fp] = layer_name
                        save_memory(memory)
                else:
                    print(f"   🔴 Test Failed: {var_or_err}")
                    
            except (ValueError, IndexError):
                print("   🔴 Invalid selection.")

        except Exception as e:
            print(f"   🔴 Error loading model: {e}")

if __name__ == "__main__":
    run_interactive_session()