import os
import json
import cv2
import random
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# --- Configuration ---
BASE_DATASET_DIR = 'observing/train'
OUTPUT_DIR = 'data/yolo_dataset'
TRAIN_RATIO = 0.7
CLASS_ID = 0
CLASS_NAME = 'target_object'

def convert_to_yolo(box, img_w, img_h):
    """
    Converts (x1, y1, x2, y2) absolute pixel coordinates to
    YOLO's (x_center_norm, y_center_norm, width_norm, height_norm) format.
    """
    try:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        img_w, img_h = float(img_w), float(img_h)

        dw = 1.0 / img_w
        dh = 1.0 / img_h
        
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        x_center_norm = x_center * dw
        y_center_norm = y_center * dh
        width_norm = width * dw
        height_norm = height * dh
        
        return (x_center_norm, y_center_norm, width_norm, height_norm)
    except Exception as e:
        print(f"Error in coordinate conversion: {e}")
        return None
    
def process_frame(cap, frame_num, boxes, video_id, split_name, img_w, img_h, output_base_dir):
    """
    Read a frame and save it along with its YOLO annotations.
    """
    
    output_images_dir = os.path.join(output_base_dir, 'images', split_name)
    output_labels_dir = os.path.join(output_base_dir, 'labels', split_name)

    # Capture the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        print(f" Warning: Frame {frame_num} of video {video_id} could not be read from video.")
        return

    # Specify file paths
    file_basename = f"{video_id}_frame_{frame_num:06d}"
    image_path = os.path.join(output_images_dir, f"{file_basename}.jpg")
    label_path = os.path.join(output_labels_dir, f"{file_basename}.txt")
    
    # Save images
    cv2.imwrite(image_path, frame)
    
    # Save YOLO annotations
    with open(label_path, 'w') as f:
        for box in boxes:
            # Convert to YOLO format
            yolo_coords = convert_to_yolo(box, img_w, img_h)
            
            if yolo_coords:
                x_c, y_c, w, h = yolo_coords
                # Write to label file
                f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
def create_yaml_file(output_dir, class_name):
    """Creates the dataset.yaml file required by YOLO."""
    
    # Get absolute paths for the YAML file
    train_path = os.path.abspath(os.path.join(output_dir, 'images', 'train'))
    val_path = os.path.abspath(os.path.join(output_dir, 'images', 'val'))
    
    yaml_content = {
        'train': train_path,
        'val': val_path,
        'nc': 1,  # number of classes
        'names': [class_name]  # list of class names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSuccessfully created {yaml_path}")
    print("This file points to your training and validation data.")
    
def build_dataset():
    annotations_file = os.path.join(BASE_DATASET_DIR, 'annotations', 'annotations.json')
    samples_dir = os.path.join(BASE_DATASET_DIR, 'samples')
    
    if not os.path.exists(annotations_file):
        print(f"Error: Annotation file not found at {annotations_file}")
        return
        
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found at {samples_dir}")
        return
        
    # Create all output directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)

    # Load all annotation records
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        try:
            all_video_records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse annotations.json. Invalid JSON: {e}")
            return
            
    if not isinstance(all_video_records, list):
         print(f"Error: Expected annotations.json to contain a list of video records.")
         return

    random.seed(42) # Use a fixed seed for reproducible splits

    print("Processing videos and splitting frames...")
    # Loop over each video file
    for record in tqdm(all_video_records, desc="Processing Videos"):
        video_id = record['video_id']
        video_path = os.path.join(samples_dir, video_id, 'drone_video.mp4')
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found, skipping: {video_path}")
            continue

        # 1. Group all annotations by frame number for this video.
        frames_to_process = {}
        for interval in record.get('annotations', []):
            for bbox_data in interval.get('bboxes', []):
                try:
                    frame_num = int(bbox_data['frame'])
                    box = (
                        int(bbox_data['x1']),
                        int(bbox_data['y1']),
                        int(bbox_data['x2']),
                        int(bbox_data['y2'])
                    )
                    
                    if frame_num not in frames_to_process:
                        frames_to_process[frame_num] = []
                    frames_to_process[frame_num].append(box)
                except KeyError as e:
                    print(f"Warning: Missing key {e} in bbox data for {video_id}, skipping box.")
                except Exception as e:
                    print(f"Warning: Error processing bbox data for {video_id}: {e}, skipping box.")
        
        if not frames_to_process:
            continue
            
        # 2. Open video and get its properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}, skipping.")
            continue
            
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if video_width == 0 or video_height == 0:
            print(f"Error: Could not get dimensions for video {video_path}, skipping.")
            cap.release()
            continue

        # 3. THIS IS THE SPLIT: Shuffle the list of frames *for this video*
        frame_items = list(frames_to_process.items())
        
        split_index = int(len(frame_items) * TRAIN_RATIO)
        train_frames = frame_items[:split_index]
        val_frames = frame_items[split_index:]

        # 4. Process and save frames to their respective splits
        for frame_num, boxes in train_frames:
            process_frame(cap, frame_num, boxes, video_id, 'train', video_width, video_height, OUTPUT_DIR)
            
        for frame_num, boxes in val_frames:
            process_frame(cap, frame_num, boxes, video_id, 'val', video_width, video_height, OUTPUT_DIR)
        
        cap.release()

    # Create the final dataset.yaml file
    create_yaml_file(OUTPUT_DIR, CLASS_NAME)
    
    print("\n--- Dataset generation complete! ---")
    print(f"Your YOLO dataset is ready in: {os.path.abspath(OUTPUT_DIR)}")
    
def main():
    build_dataset()
    
if __name__ == '__main__':
    main()