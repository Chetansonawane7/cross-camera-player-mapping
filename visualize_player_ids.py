import cv2
import json
import os
from ultralytics import YOLO

# ==== CONFIG ====
MODEL_PATH = "best.pt"
VIDEO_PATHS = {
    "broadcast": "broadcast.mp4",
    "tacticam": "tacticam.mp4"
}
OUTPUT_PATHS = {
    "broadcast": "broadcast_annotated.mp4",
    "tacticam": "tacticam_annotated.mp4"
}
MATCH_FILE = "player_matches.json"
FRAME_LIMIT = 6
PLAYER_CLASS_ID = 2

# ==== Load model and matches ====
model = YOLO(MODEL_PATH)
model.to("cpu")

with open(MATCH_FILE, "r") as f:
    player_matches = json.load(f)

# ==== Assign IDs ====
broadcast_ids = {}  # {filename: player_id}
tacticam_ids = {}   # {filename: player_id}
next_id = 0

# Assign unique ID to broadcast player crops
for idx, name in enumerate(sorted(set(m["matched_with"] for m in player_matches.values() if m["matched_with"]))):
    broadcast_ids[name] = next_id
    next_id += 1

# Transfer ID to tacticam players based on match
for tacticam_crop, match in player_matches.items():
    matched_broadcast_crop = match["matched_with"]
    if matched_broadcast_crop in broadcast_ids:
        tacticam_ids[tacticam_crop] = broadcast_ids[matched_broadcast_crop]

# ==== Draw Function ====
def draw_boxes(video_name, id_map, output_path):
    cap = cv2.VideoCapture(VIDEO_PATHS[video_name])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= FRAME_LIMIT:
            break

        # Detect
        results = model.predict(source=frame, conf=0.2, verbose=False)[0]

        player_count = 0
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != PLAYER_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop_name = f"frame_{frame_idx:04d}_obj_{player_count:02d}.jpg"

            # Check for ID
            if crop_name in id_map:
                player_id = id_map[crop_name]
                color = (0, 255, 0)
                label = f"Player {player_id}"
            else:
                color = (0, 0, 255)
                label = "Unmatched"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            player_count += 1

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Annotated video saved: {output_path}")

# ==== MAIN ====
if __name__ == "__main__":
    draw_boxes("broadcast", broadcast_ids, OUTPUT_PATHS["broadcast"])
    draw_boxes("tacticam", tacticam_ids, OUTPUT_PATHS["tacticam"])
