import os
import sys
import time
import shutil
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Tuple
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    """Centralized configuration for the box extraction process."""
    # Model Settings
    YOLO_MODEL_PATH: str = "yolo12n.pt"
    CONF_THRESHOLD: float = 0.10
    
    # Pre-processing/Cropping Settings
    PADDING: int = 20  # Padding pixels around the detected box
    USE_LARGEST_BOX: bool = True  # If False, uses the box with highest confidence
    MIN_CROP_AREA: int = 10 * 10 * 3 # Minimum required crop size (pixels * channels)

    # GrabCut Settings
    GRABCUT_ITERS: int = 5
    GRABCUT_RECT_PADDING: int = 10 # Inner padding for GrabCut initial rectangle (to avoid image edges)

    # I/O Paths
    INPUT_DIR: str = "object_images/test"
    OUTPUT_DIR: str = "debug_output_boxgrabcut"


# ==============================================================================
# 2. CORE CLASS
# ==============================================================================

class GrabCutBoxExtractor:
    """
    Combines YOLO detection with the GrabCut algorithm to robustly extract 
    a single foreground object from an image and place it on a white background.
    """
    def __init__(self, config: Config = Config):
        """
        Initializes the model and configuration.

        Args:
            config: The configuration object containing model and process settings.
        """
        self.config = config
        try:
            self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
            print(f"✅ YOLO model loaded from {self.config.YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            sys.exit(1)

    def _select_target_box(self, results: List, H: int, W: int) -> Optional[np.ndarray]:
        """Selects the single best bounding box based on configuration."""
        if not results or not results[0] or not getattr(results[0], "boxes", None):
            return None

        bxs = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        if len(bxs) == 0:
            return None
        
        if self.config.USE_LARGEST_BOX:
            # Select the box with the maximum area
            areas = (bxs[:, 2] - bxs[:, 0]) * (bxs[:, 3] - bxs[:, 1])
            idx = np.argmax(areas)
        else:
            # Select the box with the maximum confidence
            idx = np.argmax(confs)
            
        return bxs[idx]

    def _apply_grabcut(self, crop: np.ndarray) -> np.ndarray:
        """Applies the GrabCut algorithm to the cropped image."""
        
        mask = np.zeros(crop.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Initial rectangle for GrabCut
        p = self.config.GRABCUT_RECT_PADDING
        rect = (p, p, crop.shape[1] - 2 * p, crop.shape[0] - 2 * p)
        
        # Check if the rect is valid (width/height > 0)
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
             # Fallback to a tighter rect if padding is too large
             rect = (0, 0, crop.shape[1], crop.shape[0])

        cv.grabCut(
            crop, mask, rect, bgdModel, fgdModel, 
            self.config.GRABCUT_ITERS, 
            cv.GC_INIT_WITH_RECT
        )
        
        # Mask 2: Final mask (1 for sure/probable foreground, 0 otherwise)
        mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')
        
        # Apply mask to the crop
        result_fg = crop * mask2[:, :, np.newaxis]
        
        # Create a white background
        white_bg = np.ones_like(result_fg) * 255
        
        # Combine foreground with white background
        img_out = np.where(mask2[:, :, None] == 1, result_fg, white_bg)
        
        return img_out

    def process_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Performs detection, cropping, and GrabCut on a single image.

        Args:
            img: The input image (BGR format).

        Returns:
            The extracted object on a white background (BGR format), or None if detection fails.
        """
        if img is None or img.size == 0 or img.ndim != 3 or img.shape[2] != 3:
            return None

        H, W = img.shape[:2]

        # 1. Detect object with YOLO
        results = self.yolo_model.predict(img, conf=self.config.CONF_THRESHOLD, verbose=False)
        target_box = self._select_target_box(results, H, W)

        if target_box is None:
            return None

        # 2. Crop with padding
        x1, y1, x2, y2 = map(int, target_box)
        p = self.config.PADDING
        x1, y1 = max(0, x1 - p), max(0, y1 - p)
        x2, y2 = min(W, x2 + p), min(H, y2 + p)
        
        crop = img[y1:y2, x1:x2]
        
        # Check size constraints
        if crop.size < self.config.MIN_CROP_AREA:
            return None

        # 3. Apply GrabCut
        return self._apply_grabcut(crop)


# ==============================================================================
# 3. EXECUTION BLOCK
# ==============================================================================

def main_execution(config: Config = Config) -> None:
    """Handles the batch processing and file system operations."""
    
    input_dir = Path(config.INPUT_DIR)
    output_dir = Path(config.OUTPUT_DIR)

    # Setup output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"📁 Output directory created at: {output_dir}")

    processor = GrabCutBoxExtractor(config)

    # Find all image files
    valid_extensions = ["*.jpg", "*.jpeg", "*.png"]
    files: List[Path] = []
    for ext in valid_extensions:
        files.extend(input_dir.glob(ext))
    files.sort() # Ensure consistent order

    if not files:
        print(f"⚠️ No images found in input directory: {input_dir}")
        return

    print(f"🖼️ Total images found: {len(files)}")
    
    # Process batch
    st = time.time()
    succ, fail = 0, 0
    
    for f_path in files:
        img = cv.imread(str(f_path)) # Pathlib to string for OpenCV
        
        # Error handling for image read failure
        if img is None:
            print(f"❌ Fail (Read Error): {f_path.name}")
            fail += 1
            continue

        result = processor.process_image(img)
        
        if result is not None:
            output_filename = "grabcut_" + f_path.name
            cv.imwrite(str(output_dir / output_filename), result)
            succ += 1
        else:
            print(f"❌ Fail (No Detection/Small Box): {f_path.name}")
            fail += 1

    # Report results
    total_t = time.time() - st
    
    print("\n" + "="*50)
    print("✨ BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"✅ Successful Extractions: {succ}")
    print(f"💔 Failed Extractions: {fail}")
    print(f"⏱️ Total Time: {total_t:.2f}s")
    if succ > 0:
        print(f"⚡ Processing Rate (FPS): {succ/total_t:.2f}")
    print("="*50)


if __name__ == "__main__":
    main_execution(Config)