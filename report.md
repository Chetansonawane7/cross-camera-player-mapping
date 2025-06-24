# Report – Cross-Camera Player Mapping

## 1. Objective
Perform player re-identification across two video feeds (broadcast and tacticam) by assigning consistent player IDs.

## 2. Approach
- Used Ultralytics YOLOv11 (fine-tuned `best.pt`) for player detection.
- Cropped player regions from both videos for each frame.
- Extracted deep features using ResNet50.
- Used cosine similarity to match players across videos.
- Stored matches in `player_matches.json`.

## 3. Techniques Tried
- Raw pixel similarity: poor results due to background/noise.
- Histogram comparison: inconsistent results with lighting changes.
- ResNet50 features: good separation of player embeddings.
- Tried thresholds from 0.5 to 0.95 — best tradeoff at 0.8.

## 4. Challenges
- Varying number of players per frame.
- Occlusion and camera angle mismatch.
- One player sometimes matched with multiple in opposite feed.

## 5. Outcome
- Achieved reasonably accurate matching (~80–90% visually verified).
- Annotated JSON and cropped folders are saved for reference.

## 6. Limitations
- Some false positives at lower threshold.
- No temporal modeling — current method is per-frame only.

## 7. Next Steps (with more time)
- Use temporal consistency (e.g., tracking across frames).
- Incorporate color histograms + pose estimation.
- Try metric learning (e.g., triplet loss trained on soccer player re-ID).