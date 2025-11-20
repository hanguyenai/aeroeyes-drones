"""
Configuration for Advanced Detection Pipeline
TTA + Multi-Scale + WBF Ensemble (3-5 models) + SAHI
"""

import os

# ============================================================================
# PATHS
# ============================================================================

# Model weights
YOLOV11_WEIGHTS = "weights/best_yolov11s.pt"  # YOLOv11 nano/small/medium/large
YOLOV8_WEIGHTS = "weights/best_yolov8n-p2.pt"    # YOLOv8 nano/small/medium/large
# Add more model paths as needed
YOLOV10_WEIGHTS = "path/to/yolov10n.pt"  # Optional
YOLO_CUSTOM_WEIGHTS = "path/to/custom.pt"  # Optional

# Data paths
TEST_DATA_DIR = "public_test/samples"
OBJECT_IMG_DIR = "data/object_images"
PROCESSED_OBJ_DIR = "data/processed_objects"
OUTPUT_FILE = "predictions_advanced.json"

# ============================================================================
# TEST-TIME AUGMENTATION (TTA) CONFIGURATION
# ============================================================================

TTA_ENABLED = True

TTA_AUGMENTATIONS = {
    # Geometric augmentations
    'horizontal_flip': True,      # Mirror horizontally
    'vertical_flip': False,       # Mirror vertically (usually not needed)
    'rotate_90': False,           # Rotate 90° (only if objects can be rotated)
    'rotate_180': False,          # Rotate 180°
    'rotate_270': False,          # Rotate 270°
    
    # Photometric augmentations
    'brightness_variations': [0.9, 1.0, 1.1],  # Brightness multipliers
    'contrast_variations': [0.9, 1.0, 1.1],    # Contrast multipliers
}

# ============================================================================
# MULTI-SCALE CONFIGURATION
# ============================================================================

MULTI_SCALE_ENABLED = True

# Different image sizes for detection (width, height)
MULTI_SCALE_SIZES = [
    (480, 480),    # Small - fast, good for large objects
    (640, 640),    # Medium - balanced (YOLO default)
    (800, 800),    # Large - slower, better for small objects
    (1024, 1024),  # Extra Large - slowest, best for tiny objects
]

# You can adjust based on your needs:
# For speed: use only 2-3 scales
# For accuracy: use 4-5 scales
# For very small objects: add (1280, 1280)

# ============================================================================
# ENSEMBLE CONFIGURATION (3-5 MODELS)
# ============================================================================

ENSEMBLE_MODELS = [
    # Model 1: YOLOv11 (usually the best)
    {
        'name': 'YOLOv11_s',
        'path': YOLOV11_WEIGHTS,
        'imgsz': 640,              # Input image size
        'conf': 0.25,              # Confidence threshold
        'weight': 1.5              # WBF weight (higher = more important)
    },
    
    # Model 2: YOLOv8
    {
        'name': 'YOLOv8_n',
        'path': YOLOV8_WEIGHTS,
        'imgsz': 640,
        'conf': 0.25,
        'weight': 1.0
    },
    
    # Model 3: YOLOv10 (optional - uncomment if you have it)
    # {
    #     'name': 'YOLOv10',
    #     'path': YOLOV10_WEIGHTS,
    #     'imgsz': 640,
    #     'conf': 0.25,
    #     'weight': 1.2
    # },
    
    # Model 4: Custom trained model (optional)
    # {
    #     'name': 'Custom_Model',
    #     'path': YOLO_CUSTOM_WEIGHTS,
    #     'imgsz': 640,
    #     'conf': 0.25,
    #     'weight': 1.3
    # },
    
    # Model 5: Same model with different size (optional)
    # {
    #     'name': 'YOLOv11_xlarge',
    #     'path': YOLOV11_WEIGHTS,
    #     'imgsz': 1280,
    #     'conf': 0.25,
    #     'weight': 1.4
    # },
]

# ============================================================================
# SAHI CONFIGURATION (Sliced Aided Hyper Inference)
# ============================================================================

SAHI_ENABLED = True

# Slice settings
SAHI_SLICE_HEIGHT = 320         # Height of each slice (pixels)
SAHI_SLICE_WIDTH = 320          # Width of each slice (pixels)
SAHI_OVERLAP_RATIO = 0.3        # Overlap between slices (0.0-0.5)

# Recommendations:
# - For small objects: use 320x320 or 256x256 with overlap 0.3
# - For medium objects: use 640x640 with overlap 0.2
# - Higher overlap = more detections but slower

# ============================================================================
# WEIGHTED BOXES FUSION (WBF) CONFIGURATION
# ============================================================================

WBF_IOU_THRESHOLD = 0.55        # IoU threshold for fusion (0.5-0.7)
WBF_SKIP_BOX_THRESHOLD = 0.3   # Min confidence to consider
WBF_CONF_TYPE = 'box_and_model_avg'           # 'avg', 'max', or 'box_and_model_avg'

# Model weights are defined in ENSEMBLE_MODELS above

# ============================================================================
# POST-PROCESSING
# ============================================================================

# Final filtering
POST_NMS_IOU_THRESHOLD = 0.5    # Final NMS IoU threshold
POST_CONF_THRESHOLD = 0.35      # Minimum final confidence

# Soft-NMS (optional)
USE_SOFT_NMS = False
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_METHOD = 'gaussian'    # 'gaussian' or 'linear'

# ============================================================================
# PREPROCESSING
# ============================================================================

# CLAHE enhancement
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Object image preprocessing
ENABLE_REFERENCE_MATCHING = True  # Usually not needed with strong ensemble
OBJECT_PREPROCESSING_METHOD = 'color'  # 'color', 'simple', or 'none'

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# Image sizes for individual models (can override in ENSEMBLE_MODELS)
YOLOV11_IMGSZ = 640
YOLOV8_IMGSZ = 640

# Confidence thresholds (can override in ENSEMBLE_MODELS)
CONF_THRESHOLD_YOLOV11 = 0.25
CONF_THRESHOLD_YOLOV8 = 0.25

# Test-time augmentation for YOLO
ENABLE_TTA = False  # Don't use YOLO's built-in TTA, we have custom TTA

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Verbosity
YOLO_VERBOSE = False
SHOW_PROGRESS = True

# GPU
USE_GPU = True
GPU_ID = 0

# Batch processing (if supported)
BATCH_SIZE = 1  # Usually 1 for video processing

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Temporal smoothing (for video)
ENABLE_TEMPORAL_SMOOTHING = True
TEMPORAL_WINDOW_SIZE = 5
TEMPORAL_CONFIDENCE_BOOST = 0.1

# L2 normalization epsilon
L2_EPSILON = 1e-8

# ============================================================================
# PRESETS
# ============================================================================

def load_preset(preset_name: str):
    """Load predefined configuration presets"""
    global TTA_ENABLED, MULTI_SCALE_ENABLED, SAHI_ENABLED
    global POST_CONF_THRESHOLD, WBF_IOU_THRESHOLD
    
    if preset_name == 'ultra_high_accuracy':
        # Maximum accuracy, very slow
        TTA_ENABLED = True
        MULTI_SCALE_ENABLED = True
        SAHI_ENABLED = True
        TTA_AUGMENTATIONS['horizontal_flip'] = True
        TTA_AUGMENTATIONS['brightness_variations'] = [0.85, 0.9, 1.0, 1.1, 1.15]
        TTA_AUGMENTATIONS['contrast_variations'] = [0.85, 0.9, 1.0, 1.1, 1.15]
        MULTI_SCALE_SIZES.extend([(1280, 1280), (1536, 1536)])
        POST_CONF_THRESHOLD = 0.30
        WBF_IOU_THRESHOLD = 0.6
        
    elif preset_name == 'high_precision':
        # Good balance
        TTA_ENABLED = True
        MULTI_SCALE_ENABLED = True
        SAHI_ENABLED = True
        POST_CONF_THRESHOLD = 0.35
        WBF_IOU_THRESHOLD = 0.55
        
    elif preset_name == 'balanced':
        # Speed vs accuracy
        TTA_ENABLED = True
        TTA_AUGMENTATIONS['horizontal_flip'] = True
        TTA_AUGMENTATIONS['brightness_variations'] = [1.0, 1.1]
        TTA_AUGMENTATIONS['contrast_variations'] = [1.0]
        MULTI_SCALE_ENABLED = True
        MULTI_SCALE_SIZES[:] = [(640, 640), (800, 800)]
        SAHI_ENABLED = False
        POST_CONF_THRESHOLD = 0.40
        
    elif preset_name == 'fast':
        # Speed priority
        TTA_ENABLED = False
        MULTI_SCALE_ENABLED = False
        SAHI_ENABLED = False
        POST_CONF_THRESHOLD = 0.45
        # Use only 1-2 models in ENSEMBLE_MODELS
        
    elif preset_name == 'real_time':
        # Maximum speed
        TTA_ENABLED = False
        MULTI_SCALE_ENABLED = False
        SAHI_ENABLED = False
        POST_CONF_THRESHOLD = 0.50
        # Use only 1 lightweight model
        
    print(f"✓ Loaded preset: {preset_name}")

def validate_config() -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Check model files exist
    for model_cfg in ENSEMBLE_MODELS:
        if not os.path.exists(model_cfg['path']):
            errors.append(f"Model not found: {model_cfg['path']}")
    
    # Check data directory
    if not os.path.exists(TEST_DATA_DIR):
        errors.append(f"Test data directory not found: {TEST_DATA_DIR}")
    
    # Check thresholds
    if not (0.0 <= POST_CONF_THRESHOLD <= 1.0):
        errors.append(f"Invalid POST_CONF_THRESHOLD: {POST_CONF_THRESHOLD}")
    
    if not (0.0 <= WBF_IOU_THRESHOLD <= 1.0):
        errors.append(f"Invalid WBF_IOU_THRESHOLD: {WBF_IOU_THRESHOLD}")
    
    # Check ensemble models
    if len(ENSEMBLE_MODELS) < 1:
        errors.append("At least 1 model required in ENSEMBLE_MODELS")
    
    if len(ENSEMBLE_MODELS) > 5:
        print("⚠ Warning: Using more than 5 models may be very slow")
    
    # Print errors
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    return True

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: Maximum Accuracy (Competition Mode)
config.load_preset('ultra_high_accuracy')
run_advanced_inference()

# Example 2: Balanced (Default)
config.load_preset('high_precision')
run_advanced_inference()

# Example 3: Fast Inference
config.load_preset('fast')
run_advanced_inference()

# Example 4: Custom Configuration
TTA_ENABLED = True
TTA_AUGMENTATIONS['horizontal_flip'] = True
MULTI_SCALE_ENABLED = True
MULTI_SCALE_SIZES = [(640, 640), (800, 800)]
SAHI_ENABLED = True
POST_CONF_THRESHOLD = 0.35

ENSEMBLE_MODELS = [
    {'name': 'YOLOv11', 'path': 'yolov11n.pt', 'imgsz': 640, 'conf': 0.25, 'weight': 1.5},
    {'name': 'YOLOv8', 'path': 'yolov8n.pt', 'imgsz': 640, 'conf': 0.25, 'weight': 1.0},
    {'name': 'Custom', 'path': 'custom.pt', 'imgsz': 640, 'conf': 0.25, 'weight': 1.2},
]

run_advanced_inference()
"""