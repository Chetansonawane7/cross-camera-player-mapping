import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
CROP_DIR = "crops"  
OUTPUT_DIR = "features"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224  

# ==== PREPROCESSING ====
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

# ==== LOAD MODEL ====
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove final classification layer
resnet.to(DEVICE)
resnet.eval()
print(f"✅ ResNet50 loaded on {DEVICE}")

# ==== FEATURE EXTRACT FUNCTION ====
def extract_features_from_folder(folder_name):
    folder_path = os.path.join(CROP_DIR, folder_name)
    output_features = {}

    for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
        if not img_name.endswith(".jpg"):
            continue

        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = resnet(img_tensor).squeeze().cpu().numpy()  # shape: (2048,)
            output_features[img_name] = embedding

    # Save as .npy
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, f"{folder_name}_features.npy"), output_features)
    print(f"💾 Saved: {folder_name}_features.npy")

# ==== MAIN ====
if __name__ == "__main__":
    extract_features_from_folder("broadcast")
    extract_features_from_folder("tacticam")
