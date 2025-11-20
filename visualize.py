#!/usr/bin/env python3
import os, json, cv2
from collections import defaultdict
from tqdm import tqdm

# ==== CONFIG (đổi theo cấu trúc của bạn) ====
PRED_JSON     = "submission.json"
TEST_DATA_DIR = "public_test/samples"       # thư mục chứa <video_id>/drone_video.mp4
VIDEO_NAME    = "drone_video.mp4"
OUT_DIR       = "viz_out"
BOX_THICKNESS = 2
BOX_COLOR     = (0, 255, 0)   # BGR
TEXT_COLOR    = (255, 255, 255)

# ---------- helpers ----------
def is_normalized_xyxy(b):
    # x1,y1,x2,y2 trong [0,1]? (chịu lỗi nhỏ)
    keys = ["x1","y1","x2","y2"]
    if not all(k in b for k in keys): return False
    vals = [float(b[k]) for k in keys]
    return all(0.0 <= v <= 1.001 for v in vals)

def is_xyxy(b):
    return all(k in b for k in ["x1","y1","x2","y2"])

def is_normalized_xywh(b):
    # xc,yc,w,h trong [0,1]?
    for ks in (["xc","yc","w","h"], ["x_center","y_center","width","height"]):
        if all(k in b for k in ks):
            vals = [float(b[k]) for k in ks]
            if all(0.0 <= v <= 1.001 for v in vals):
                return ks
    return None

def xywh_to_xyxy(xc, yc, w, h, W, H):
    x1 = int(round((xc - w/2) * W))
    y1 = int(round((yc - h/2) * H))
    x2 = int(round((xc + w/2) * W))
    y2 = int(round((yc + h/2) * H))
    # clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    return x1, y1, x2, y2

def denormalize_xyxy(x1, y1, x2, y2, W, H):
    X1 = int(round(x1 * W));  Y1 = int(round(y1 * H))
    X2 = int(round(x2 * W));  Y2 = int(round(y2 * H))
    X1, Y1 = max(0, X1), max(0, Y1)
    X2, Y2 = min(W-1, X2), min(H-1, Y2)
    return X1, Y1, X2, Y2

def load_predictions(json_path):
    """
    Trả về dict:
      preds[video_id][frame_idx] = list of bbox dicts (giữ nguyên keys gốc để xử lý linh hoạt)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    preds = {}
    for vid in data:
        video_id = vid["video_id"]
        bboxes_by_frame = defaultdict(list)

        for det in vid.get("detections", []):
            for bb in det.get("bboxes", []):
                if "frame" not in bb:
                    continue
                bboxes_by_frame[int(bb["frame"])].append(bb)
        preds[video_id] = bboxes_by_frame
    return preds

def resolve_bbox_xyxy(bb, W, H):
    """
    Từ 1 bbox object trong JSON -> (x1,y1,x2,y2) pixel.
    Hỗ trợ:
      - xyxy normalized (0..1)  -> denormalize
      - xyxy pixel              -> dùng trực tiếp
      - xywh normalized (0..1)  -> convert + denormalize
    """
    if is_xyxy(bb):
        x1, y1, x2, y2 = float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])
        if is_normalized_xyxy(bb):
            return denormalize_xyxy(x1, y1, x2, y2, W, H)
        # assume pixel
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    ks = is_normalized_xywh(bb)
    if ks:
        xc, yc, w, h = (float(bb[ks[0]]), float(bb[ks[1]]),
                        float(bb[ks[2]]), float(bb[ks[3]]))
        return xywh_to_xyxy(xc, yc, w, h, W, H)

    # Không nhận dạng được format -> bỏ qua
    return None

def visualize_video(video_path, bboxes_by_frame, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total, desc=os.path.basename(video_path), leave=False)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        for bb in bboxes_by_frame.get(frame_idx, []):
            xyxy = resolve_bbox_xyxy(bb, W, H)
            if xyxy is None: continue
            x1,y1,x2,y2 = xyxy
            cv2.rectangle(frame, (x1,y1), (x2,y2), BOX_COLOR, BOX_THICKNESS)

            # vẽ text nếu có class/score
            label_parts = []
            if "cls" in bb:   label_parts.append(str(bb["cls"]))
            if "score" in bb: label_parts.append(f"{float(bb['score']):.2f}")
            if label_parts:
                cv2.putText(frame, " ".join(label_parts),
                            (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

def main():
    preds = load_predictions(PRED_JSON)
    os.makedirs(OUT_DIR, exist_ok=True)

    for video_id, by_frame in preds.items():
        video_path = os.path.join(TEST_DATA_DIR, video_id, VIDEO_NAME)
        if not os.path.exists(video_path):
            print(f"[WARN] Not found: {video_path}")
            continue
        out_path = os.path.join(OUT_DIR, f"{video_id}_annotated.mp4")
        visualize_video(video_path, by_frame, out_path)

    print(f"✅ Done. Annotated videos are in: {OUT_DIR}")

if __name__ == "__main__":
    main()
