import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
import pyiqa
import csv

# ==========================================================
# 🔧 PATH — CHANGE THIS TO YOUR ENHANCED IMAGE FOLDER
# ==========================================================
pred_dir = r"C:\Users\Admin\Desktop\Results\DCAI\VV\VV"
save_csv = True

# ==========================================================
# ⭐ Load REAL NIQE & BRISQUE (pyiqa)
# ==========================================================
print("🚀 Loading REAL NIQE and BRISQUE (pyiqa)...")

device = "cuda" if torch.cuda.is_available() else "cpu"

niqe_metric = pyiqa.create_metric("niqe", device=device)
brisque_metric = pyiqa.create_metric("brisque", device=device)

to_tensor = transforms.ToTensor()

# ==========================================================
# 🧮 Metric Functions
# ==========================================================
def compute_niqe(img):
    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    return float(niqe_metric(img_tensor).item())

def compute_brisque(img):
    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    return float(brisque_metric(img_tensor).item())

# ==========================================================
# 🔄 Evaluate Metrics
# ==========================================================
niqe_scores, brisque_scores, pi_scores = [], [], []
results = []

print(f"\n🔍 Evaluating images in: {pred_dir}\n")

files = sorted([
    f for f in os.listdir(pred_dir)
    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
])

print(f"✅ Found {len(files)} images.\n")

for filename in tqdm(files):
    img_path = os.path.join(pred_dir, filename)

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Could not open {filename}: {e}")
        continue

    try:
        n_score = compute_niqe(img)
        b_score = compute_brisque(img)
    except Exception as e:
        print(f"⚠️ Metric error for {filename}: {e}")
        continue

    pi_score = (n_score + b_score) / 2

    niqe_scores.append(n_score)
    brisque_scores.append(b_score)
    pi_scores.append(pi_score)

    results.append({
        "image": filename,
        "NIQE": n_score,
        "BRISQUE": b_score,
        "PI": pi_score
    })

# ==========================================================
# 📊 SUMMARY
# ==========================================================
def safe_mean(x):
    return np.mean(x) if len(x) > 0 else np.nan

print("\n===== REAL No-Reference Evaluation Results =====")
print(f"Images Evaluated : {len(results)}")
print(f"NIQE   : {safe_mean(niqe_scores):.4f}")
print(f"BRISQUE: {safe_mean(brisque_scores):.4f}")
print(f"PI     : {safe_mean(pi_scores):.4f}")
print("\n✅ Lower values = better perceptual quality.")

# ==========================================================
# 💾 SAVE CSV
# ==========================================================
if save_csv:
    csv_path = os.path.join(pred_dir, "metrics_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "NIQE", "BRISQUE", "PI"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📁 Saved results to: {csv_path}")