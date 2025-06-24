import cv2
import os
from ultralytics import YOLO
import torch

# ==== CONFIG ====
VIDEO_PATHS = {
    "broadcast": "broadcast.mp4",
    "tacticam": "tacticam.mp4"
}
MODEL_PATH = "best.pt"  # ✅ Everything in same folder
CROP_DIR = "crops"
CROP_SIZE = 224  # Resize for ResNet input

# ==== LOAD MODEL ====
model = YOLO(MODEL_PATH)
model.to("cpu")
print(f"✅ Model Loaded: {MODEL_PATH}")
print("📦 Classes:", model.model.names)  # This tells you what class IDs mean

# Create output folders
os.makedirs(f"{CROP_DIR}/broadcast", exist_ok=True)
os.makedirs(f"{CROP_DIR}/tacticam", exist_ok=True)

# ==== PROCESS VIDEO FUNCTION ====
def process_video(video_name):
    print(f"\n🔍 Processing {video_name}.mp4")
    cap = cv2.VideoCapture(VIDEO_PATHS[video_name])
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > 5:  # Debug only 5 frames
            break

        # Run YOLO with lower confidence threshold
        results = model.predict(source=frame, conf=0.2, verbose=False)[0]
        print(f"[{video_name}] Frame {frame_idx} - {len(results.boxes)} boxes")

        player_count = 0
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            print(f" ➤ Detected class ID: {cls_id}")

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            filename = f"{CROP_DIR}/{video_name}/frame_{frame_idx:04d}_obj_{player_count:02d}.jpg"
            cv2.imwrite(filename, crop)
            player_count += 1

            # Draw box for debug
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(cls_id), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow(f"{video_name} - detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done: {video_name} — {frame_idx} frames.")

# ==== MAIN ====
if __name__ == "__main__":
    process_video("broadcast")
    process_video("tacticam")
