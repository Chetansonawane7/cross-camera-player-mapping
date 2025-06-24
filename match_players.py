import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# ==== CONFIG ====
FEATURE_DIR = "features"
THRESHOLD = 0.8  # similarity threshold to consider as same player

# ==== LOAD FEATURES ====
broadcast_feats = np.load(os.path.join(FEATURE_DIR, "broadcast_features.npy"), allow_pickle=True).item()
tacticam_feats = np.load(os.path.join(FEATURE_DIR, "tacticam_features.npy"), allow_pickle=True).item()

# ==== MATCHING ====
matches = {}

# Convert feature dicts to lists
broadcast_names, broadcast_vectors = zip(*broadcast_feats.items())
tacticam_names, tacticam_vectors = zip(*tacticam_feats.items())

broadcast_vectors = np.array(broadcast_vectors)
tacticam_vectors = np.array(tacticam_vectors)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tacticam_vectors, broadcast_vectors)

# Find best match for each tacticam player
for i, tacticam_name in enumerate(tacticam_names):
    best_match_idx = np.argmax(similarity_matrix[i])
    best_score = similarity_matrix[i][best_match_idx]

    if best_score >= THRESHOLD:
        matches[tacticam_name] = {
            "matched_with": broadcast_names[best_match_idx],
            "similarity": float(best_score)
        }
    else:
        matches[tacticam_name] = {
            "matched_with": None,
            "similarity": float(best_score)
        }

# ==== SAVE RESULTS ====
with open("player_matches.json", "w") as f:
    json.dump(matches, f, indent=2)

print(f"✅ Matching completed! Saved to player_matches.json")
