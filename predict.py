# ==============================================================================
# predict.py - AEROEYES DRONE DETECTION & TRACKING SYSTEM
# YOLO + D-FINE + MOBILENET-V3 + KALMAN + CLAHE + TTA + SMART SKIP
# ==============================================================================

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO
import gc
from typing import List, Dict, Union, Tuple, Optional

# --- Required Dependencies ---
try:
    from ensemble_boxes import weighted_boxes_fusion
    from filterpy.kalman import KalmanFilter
except ImportError:
    raise ImportError("Error: Please install 'pip install ensemble-boxes filterpy'")

# --- D-FINE IMPORT SETUP ---
# Ensure this path matches the Docker file structure /code/D-FINE
DFINE_DIR = str(Path(os.getcwd()) / 'D-FINE')
if DFINE_DIR not in sys.path:
    sys.path.append(DFINE_DIR)
try:
    from src.core import YAMLConfig
except ImportError as e:
    print(f"❌ D-Fine import Error. Ensure the D-FINE folder is copied to /code: {e}")
    YAMLConfig = None


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    """Centralized configuration for the entire system."""
    # Model Paths
    YOLOV8_PATH: str = 'weights/last_yolov8s-p2.pt'
    YOLOV11_PATH: str = 'weights/last_yolov11p2.pt'
    DFINE_CONFIG_PATH: str = str(Path(DFINE_DIR) / 'configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml')
    DFINE_WEIGHTS_PATH: str = str(Path(DFINE_DIR) / 'output/dfine_hgnetv2_s_obj2custom/best_stg2.pth')
    
    # Thresholds
    CONF_YOLOV8: float = 0.15
    CONF_YOLOV11: float = 0.15
    CONF_DFINE: float = 0.15
    WBF_IOU_THR: float = 0.5
    WBF_WEIGHTS: List[float] = [1.0, 1.0, 1.2]
    
    # Matching & Geometric
    SIM_THRESHOLD: float = 0.25
    SIM_WEIGHT: float = 0.6
    CONF_WEIGHT: float = 0.4
    MIN_BOX_AREA: int = 100
    MAX_BOX_AREA_RATIO: float = 0.4
    
    # Tracking & Optimization
    USE_KALMAN: bool = True
    USE_SMART_SKIP: bool = True
    SKIP_INTERVAL: int = 2
    
    # Preprocessing
    USE_CLAHE: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)
    
    # Test Time Augmentation (TTA)
    USE_TTA: bool = True 

    # File Paths
    TEST_DATA_DIR: str = 'public_test/samples'  # Standard path in Docker
    OUTPUT_FILE: str = 'submission.json' # Output to /result folder
    GRABCUT_ROOT: str = 'box_grabcut' # Folder containing GrabCut reference images


# ==============================================================================
# 2. CORE UTILITIES
# ==============================================================================

class ImagePreprocessor:
    """Image processing utilities: CLAHE and TTA."""
    def __init__(self):
        self.clahe = None
        if Config.USE_CLAHE:
            self.clahe = cv2.createCLAHE(
                clipLimit=Config.CLAHE_CLIP_LIMIT, 
                tileGridSize=Config.CLAHE_TILE_SIZE
            )

    def apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Applies CLAHE to the L channel (Lightness) of the LAB color space."""
        if not Config.USE_CLAHE: return frame_bgr
        
        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception:
            return frame_bgr

    def get_tta_batch(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generates a batch of images for TTA (horizontal flip)."""
        frames = [frame]
        transforms = [{'flip': False}]
        
        if Config.USE_TTA:
            frames.append(cv2.flip(frame, 1))
            transforms.append({'flip': True, 'width': frame.shape[1]})
            
        return frames, transforms

    def deaugment_boxes(self, boxes: np.ndarray, transform: Dict) -> np.ndarray:
        """De-augments boxes from the TTA (flipped) image back to original coordinates."""
        if len(boxes) == 0 or not transform['flip']:
            return boxes
            
        W = transform['width']
        boxes_copy = boxes.copy()
        # Flip coordinates: x1_new = W - x2_old, x2_new = W - x1_old
        boxes_copy[:, 0] = W - boxes[:, 2]
        boxes_copy[:, 2] = W - boxes[:, 0]
        return boxes_copy


class DFineWrapper:
    """Wrapper for the D-FINE (DETR-like) model."""
    def __init__(self, config_path: str, weights_path: str, device: str = 'cuda'):
        self.device = device
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[T.Compose] = None
        
        if YAMLConfig is None: return
        if not Path(config_path).exists() or not Path(weights_path).exists(): return
        
        try:
            # Load config and build model
            cfg = YAMLConfig(config_path, resume=weights_path)
            model = cfg.model if hasattr(cfg, 'model') and cfg.model else cfg.build_model()
            
            # Load weights
            if hasattr(model, 'deploy'): model = model.deploy()
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = checkpoint.get('ema', {}).get('module', checkpoint.get('model', checkpoint))
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(new_state_dict, strict=False)
            self.model = model.to(device).eval()
            
            # D-Fine required Transform
            self.transform = T.Compose([
                T.ToPILImage(), T.Resize((640, 640)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print("✅ D-FINE loaded successfully.")
        except Exception as e:
            self.model = None
            print(f"⚠️ D-FINE Load Error: {e}")
    
    def predict(self, frame_bgr: np.ndarray, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs inference for D-FINE."""
        if self.model is None or self.transform is None: return np.array([]), np.array([]), np.array([])
        
        try:
            H, W = frame_bgr.shape[:2]
            # Convert BGR -> RGB and apply Transform
            blob = self.transform(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(blob)
                
                # Handle output format (dict or tuple)
                if isinstance(out, dict): 
                    p_boxes, p_logits = out['pred_boxes'][0], out['pred_logits'][0]
                else: 
                    # Assuming out = (boxes, logits, ...)
                    p_boxes, p_logits = out[0][0], out[1][0] 
                
                scores, labels = p_logits.sigmoid().max(-1)[0], p_logits.sigmoid().argmax(-1)
                mask = scores > conf_threshold
                if not mask.any(): return np.array([]), np.array([]), np.array([])
                
                vb, vs, vl = p_boxes[mask].cpu().numpy(), scores[mask].cpu().numpy(), labels[mask].cpu().numpy()
                
                # Normalize -> Pixel coordinates (x_min, y_min, x_max, y_max)
                cx, cy, w, h = vb.T
                boxes = np.stack([(cx-0.5*w)*W, (cy-0.5*h)*H, (cx+0.5*w)*W, (cy+0.5*h)*H], axis=1)
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)
                return boxes, vs, vl
        except Exception: 
            return np.array([]), np.array([]), np.array([])


class KalmanTracker:
    """8D Kalman Filter (x, y, w, h, vx, vy, vw, vh) for object tracking."""
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # Transition matrix (F): 4 positions + 4 velocities
        self.kf.F = np.eye(8); self.kf.F[0,4]=1; self.kf.F[1,5]=1; self.kf.F[2,6]=1; self.kf.F[3,7]=1
        # Measurement matrix (H): only measuring 4 positions
        self.kf.H = np.eye(4, 8)
        # Tuning parameters
        self.kf.R *= 10; self.kf.P *= 1000; self.kf.Q *= 0.01
        self.initialized: bool = False
        self.missed: int = 0

    def initialize(self, bbox: Union[List[float], np.ndarray]) -> None:
        """Initializes the state."""
        x, y, w, h = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(8, 1)
        self.initialized = True
        self.missed = 0
        
    def predict(self) -> Optional[List[float]]:
        """Predicts the next state."""
        if not self.initialized: return None
        self.kf.predict()
        x, y, w, h = self.kf.x[:4].flatten()
        return [float(x-w/2), float(y-h/2), float(x+w/2), float(y+h/2)]
    
    def update(self, bbox: Union[List[float], np.ndarray]) -> None:
        """Updates the state with a new measurement."""
        if not self.initialized: self.initialize(bbox); return
        x, y, w, h = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf.update([x, y, w, h]); self.missed = 0

    def get_bbox(self) -> Optional[List[float]]:
        """Returns the current bounding box from the state."""
        if not self.initialized: return None
        x, y, w, h = self.kf.x[:4].flatten()
        return [float(x-w/2), float(y-h/2), float(x+w/2), float(y+h/2)]
    
    def mark_missed(self) -> None: self.missed += 1
    def is_lost(self) -> bool: return self.missed > 10


class FeatureExtractor:
    """Feature extraction using MobileNet-V3 Small for Similarity Matching."""
    def __init__(self, device: str):
        self.device = device
        # Load pre-trained MobileNet-V3 Small
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Replace classifier layer with Identity to get the feature vector
        self.model.classifier = torch.nn.Identity()
        self.model.to(device).eval()
        
        # Standard Transform for MobileNet
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)), 
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_emb: Optional[torch.Tensor] = None

    def load_reference(self, ref_paths: List[Path]) -> None:
        """Extracts and averages features from GrabCut reference images."""
        tensors = []
        for p in ref_paths:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None: continue

            # Handle Transparency (for 4-channel PNGs from GrabCut)
            if img.ndim == 3 and img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
                img_rgb = cv2.merge((b, g, r))
                # Set transparent regions (a=0) to black background
                img_rgb[a == 0] = [0, 0, 0]
                img = img_rgb
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img))

        if not tensors:
            self.target_emb = None
            return
        
        with torch.no_grad():
            batch = torch.stack(tensors).to(self.device)
            # Average the vectors and normalize
            emb = self.model(batch).mean(0, keepdim=True)
            self.target_emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> Optional[torch.Tensor]:
        """Extracts features from the bounding box crops in the current frame."""
        if len(boxes) == 0: return None
        crops = []
        H, W = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(self.transform(crop_rgb))
            
        if not crops: return None
        
        with torch.no_grad():
            batch = torch.stack(crops).to(self.device)
            return torch.nn.functional.normalize(self.model(batch), p=2, dim=1)


# ==============================================================================
# 3. ENSEMBLE AND POST-PROCESSING
# ==============================================================================

def wbf_ensemble(boxes_list: List[np.ndarray], scores_list: List[np.ndarray], labels_list: List[np.ndarray], W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Weighted Boxes Fusion (WBF) to combine predictions."""
    norm_boxes = []
    for boxes in boxes_list:
        if len(boxes) > 0:
            b = boxes.copy()
            b[:, [0, 2]] /= W; b[:, [1, 3]] /= H
            norm_boxes.append(b.tolist())
        else: norm_boxes.append([])
    
    try:
        # labels_list is ignored by WBF in this context as we assume a single class
        wb, ws, _ = weighted_boxes_fusion(norm_boxes, [s.tolist() for s in scores_list], [l.tolist() for l in labels_list], weights=None, iou_thr=Config.WBF_IOU_THR)
        if len(wb) > 0:
            wb[:, [0, 2]] *= W; wb[:, [1, 3]] *= H
            return wb, np.array(ws)
    except Exception: 
        pass
    
    return np.array([]), np.array([])

def geometric_filter(boxes: np.ndarray, scores: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """Filters boxes based on minimum and maximum allowed area."""
    if len(boxes) == 0: return boxes, scores
    areas = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    valid = (areas >= Config.MIN_BOX_AREA) & (areas <= W * H * Config.MAX_BOX_AREA_RATIO)
    return boxes[valid], scores[valid]


# ==============================================================================
# 4. MAIN PREDICTOR CLASS (For clean batch processing and Jupyter)
# ==============================================================================

class AeroEyesPredictor:
    """The main class for handling video processing, detection, and tracking."""
    def __init__(self, config=Config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing on {self.device}")
        
        self.yolo8 = YOLO(self.config.YOLOV8_PATH)
        self.yolo11 = YOLO(self.config.YOLOV11_PATH)
        self.dfine = DFineWrapper(self.config.DFINE_CONFIG_PATH, self.config.DFINE_WEIGHTS_PATH, self.device)
        self.extractor = FeatureExtractor(self.device)
        self.preprocessor = ImagePreprocessor()
        self.tracker: KalmanTracker = KalmanTracker()
        self.target_emb: Optional[torch.Tensor] = None
        self.current_video_id: str = ""
        
        if self.dfine.model is None:
             print("⚠️ Warning: D-FINE model failed to load. Running with YOLO only.")
        
        print("✅ AeroEyesPredictor initialized. All models loaded.")

    def load_reference(self, class_name: str) -> None:
        """
        Loads and encodes the reference feature vector for a new video class.
        This must be called once per video.
        """
        self.tracker = KalmanTracker() # Reset tracker for new video
        
        # Path to reference images (GrabCut outputs)
        ref_dir = Path(self.config.GRABCUT_ROOT) / class_name
        
        if not ref_dir.exists():
            print(f"⚠️ Warning: Reference directory for '{class_name}' not found at {ref_dir}. Skipping feature matching.")
            self.target_emb = None
            return

        ref_imgs = [p for p in ref_dir.glob('*.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        if len(ref_imgs) > 0:
            self.extractor.load_reference(ref_imgs)
            self.target_emb = self.extractor.target_emb
            print(f"✅ Loaded {len(ref_imgs)} reference images for '{class_name}'.")
        else:
            self.target_emb = None
            print(f"⚠️ Warning: No reference images found in {ref_dir}. Skipping feature matching.")

    def predict_streaming(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[List[int]]:
        """
        Processes a single frame in a streaming fashion.
        Returns the final clipped bounding box (x1, y1, x2, y2) or None.
        """
        H, W = frame_bgr.shape[:2]
        
        # === STEP 1: KALMAN PREDICT ===
        pred_box = self.tracker.predict()
        
        should_detect = True
        if self.config.USE_SMART_SKIP and frame_idx % self.config.SKIP_INTERVAL != 0 and self.tracker.initialized:
            should_detect = False
        
        best_box: Optional[np.ndarray] = None
        
        if should_detect:
            # --- STEP 2: Preprocessing & TTA ---
            frame_proc = self.preprocessor.apply_clahe(frame_bgr)
            tta_frames, tta_transforms = self.preprocessor.get_tta_batch(frame_proc)
            
            all_boxes, all_scores, all_labels = [], [], []
            
            # --- STEP 3: Multi-Model Inference (with TTA) ---
            for t_frame, transform in zip(tta_frames, tta_transforms):
                # YOLOv8 Inference
                r8 = self.yolo8.predict(t_frame, imgsz=640, conf=self.config.CONF_YOLOV8, verbose=False)[0]
                if r8.boxes:
                    b = r8.boxes.xyxy.cpu().numpy()
                    b = self.preprocessor.deaugment_boxes(b, transform)
                    all_boxes.append(b); all_scores.append(r8.boxes.conf.cpu().numpy()); all_labels.append(r8.boxes.cls.cpu().numpy())

                # YOLOv11 Inference
                r11 = self.yolo11.predict(t_frame, imgsz=640, conf=self.config.CONF_YOLOV11, verbose=False)[0]
                if r11.boxes:
                    b = r11.boxes.xyxy.cpu().numpy()
                    b = self.preprocessor.deaugment_boxes(b, transform)
                    all_boxes.append(b); all_scores.append(r11.boxes.conf.cpu().numpy()); all_labels.append(r11.boxes.cls.cpu().numpy())

                # D-Fine Inference
                bd, sd, ld = self.dfine.predict(t_frame, conf_threshold=self.config.CONF_DFINE)
                if len(bd) > 0:
                    bd = self.preprocessor.deaugment_boxes(bd, transform)
                    all_boxes.append(bd); all_scores.append(sd); all_labels.append(ld)

            # --- STEP 4: WBF Ensemble & Geometric Filter ---
            if not all_boxes:
                wb, ws = np.array([]), np.array([])
            else:
                wb, ws = wbf_ensemble(all_boxes, all_scores, all_labels, W, H)
            
            if len(wb) > 0:
                wb, ws = geometric_filter(wb, ws, W, H)
                
                # --- STEP 5: Feature Matching with MobileNet ---
                if len(wb) > 0:
                    if self.target_emb is not None:
                        feats = self.extractor.extract(frame_proc, wb)
                        if feats is not None:
                            # Cosine Similarity
                            sims = torch.mm(feats, self.target_emb.T).squeeze(1).cpu().numpy()
                            
                            best_score_val = -1
                            for i, sim in enumerate(sims):
                                if i >= len(ws): break
                                
                                # Combined Score: Similarity * Weight + Confidence * Weight
                                combined = self.config.SIM_WEIGHT * sim + self.config.CONF_WEIGHT * float(ws[i])
                                
                                if sim > self.config.SIM_THRESHOLD and combined > best_score_val:
                                    best_score_val = combined
                                    best_box = wb[i]
                    
                    if best_box is None and len(ws) > 0:
                        # Fallback: take the highest confidence box if no match or no reference
                        idx_max = np.argmax(ws)
                        best_box = wb[idx_max]

        # === STEP 6: TRACKER UPDATE ===
        final_box: Optional[List[float]] = None
        if best_box is not None:
            self.tracker.update(best_box)
            final_box = self.tracker.get_bbox()
        else:
            self.tracker.mark_missed()
            if not self.tracker.is_lost():
                final_box = pred_box
        
        # --- STEP 7: Format and Clip Output ---
        if final_box is not None:
            bx = list(map(int, final_box))
            # Clip box to image boundaries
            bx[0] = max(0, bx[0]); bx[1] = max(0, bx[1])
            bx[2] = min(W, bx[2]); bx[3] = min(H, bx[3])
            
            if bx[2] > bx[0] and bx[3] > bx[1]:
                return bx # Return the final box as [x1, y1, x2, y2]
        
        return None # Return None if no box is found or tracked


# ==============================================================================
# 5. EXECUTION BLOCK
# ==============================================================================

def run_batch_inference(predictor: AeroEyesPredictor) -> None:
    """Main execution function to process all videos in the test directory."""
    
    # Process Videos
    test_data_path = Path(predictor.config.TEST_DATA_DIR)
    video_folders = sorted([p for p in test_data_path.iterdir() if p.is_dir()])
    
    # Ensure output directory exists
    Path("/result").mkdir(exist_ok=True)
    all_preds = []
    
    for folder_path in tqdm(video_folders, desc="Processing Videos"):
        folder_name = folder_path.name
        vid_path = folder_path / "drone_video.mp4"
        
        if not vid_path.exists(): continue
        
        # Load reference and reset tracker (Calls load_reference inside the predictor)
        class_name = folder_name.split('_')[0] 
        predictor.load_reference(class_name)

        cap = cv2.VideoCapture(str(vid_path))
        video_bboxes = []
        idx = 0
        
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            
            # Use the streaming prediction method
            final_box = predictor.predict_streaming(frame_bgr, idx)
            
            if final_box is not None:
                video_bboxes.append({
                    "frame": idx, 
                    "x1": final_box[0], 
                    "y1": final_box[1], 
                    "x2": final_box[2], 
                    "y2": final_box[3]
                })
            
            idx += 1
            
        cap.release()
        all_preds.append({"video_id": folder_name, "detections": [{"bboxes": video_bboxes}] if video_bboxes else []})

    # Save results
    output_path = Path(predictor.config.OUTPUT_FILE)
    with open(output_path, 'w') as f: 
        json.dump(all_preds, f, indent=4)
        
    print(f"✅ Batch inference complete. Results saved to {output_path}")

if __name__ == '__main__':
    # The standard entry point for predict.py
    predictor = AeroEyesPredictor(Config)
    run_batch_inference(predictor)