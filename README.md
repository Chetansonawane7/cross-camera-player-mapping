# Cross-Camera Player Mapping 

## 👤 Author
Chetan Sonawane  

## 🎯 Task: Cross-Camera Player Mapping
Identify and consistently label players across two different camera feeds (`broadcast.mp4` and `tacticam.mp4`) of the same gameplay.

## 📁 Folder Structure
Cross-Camera-Player-Mapping/
│
├── best.pt                          # Provided YOLOv11 model
├── broadcast.mp4                    # Input video (camera 1)
├── tacticam.mp4                     # Input video (camera 2)
├── broadcast_annotated.mp4          # Final annotated video with IDs
├── tacticam_annotated.mp4           # Final annotated video with IDs
│
├── crops/                           # Cropped player images
│   ├── broadcast/
│   └── tacticam/
│
├── features/                        # ResNet50 feature vectors
│   ├── broadcast_features.npy
│   └── tacticam_features.npy
│
├── player_matches.json              # Final player ID mapping across videos
│
├── main.py                          # Step 1: Detection and cropping
├── extract_features.py              # Step 2: ResNet50 embeddings
├── match_players.py                 # Step 3: Cosine similarity-based matching
├── visualize_player_ids.py          # Step 4: Draw IDs and render final video
│
├── README.md                        # Setup instructions, dependencies, how to run
├── report.md                        # Brief report of approach, methods, and challenges
├── requirements.txt 

## ⚙️ Setup & Dependencies

Tested with:
- Python 3.9
- torch >= 2.1.0
- torchvision
- ultralytics
- opencv-python
- numpy
- tqdm
- scipy

Install all dependencies:
```bash
pip install -r requirements.txt
```

🚀 How to Run:
1. Detect and Crop Players:
   ```bash
   python main.py
   ```
2. Extract Embeddings:
   ```bash
   python extract_features.py
   ```
3. Match Players Between Videos:
   ```bash
   python match_players.py
   ```
4. Visualize Annotated Videos:
   ```bash
   python visualize_player_ids.py
   ```

📦 Output
player_matches.json: maps players between camera feeds
broadcast_annotated.mp4 and tacticam_annotated.mp4: videos with consistent IDs across views
Cropped player patches saved in crops/
Feature embeddings saved in features/

📌 Notes
A fixed cosine similarity threshold (0.8) was used for matching.
Some mismatches may occur due to occlusion, similar appearances, or lack of tracking.
Can be extended with global optimization (Hungarian algorithm), temporal smoothing, or tracking.

Let me know if you also want a `.pdf` version of the report or help generating `requirements.txt`. You're now fully ready to submit!
