import os
import torch
import lpips
import numpy as np
from piq import FID
from piq.feature_extractors import InceptionV3
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================================
# 🔧 PATHS
# ==========================================================
pred_dir = r"C:\Users\Admin\Desktop\Results\LYT-Net\LOLv2-Syn\Low"
gt_dir   = r"C:\Users\Admin\Desktop\Datasets\LOLv2\Synthetic\test\Normal"

# ==========================================================
# ⚙️ DEVICE & METRICS
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)
fid_metric = FID().to(device)

psnr_list, ssim_list, lpips_list = [], [], []

# ==========================================================
# 🔄 Helpers
# ==========================================================
def to_tensor(img):
    arr = np.asarray(img, np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

def list_images(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

# ==========================================================
# 📂 Load & sort files
# ==========================================================
pred_files = list_images(pred_dir)
gt_files   = list_images(gt_dir)

assert len(pred_files) == len(gt_files), \
    f"Pred ({len(pred_files)}) and GT ({len(gt_files)}) counts differ!"

print(f"\n✅ Using ORDER-BASED pairing ({len(pred_files)} image pairs)\n")

# ==========================================================
# 🧮 PSNR / SSIM / LPIPS (ORDER-BASED)
# ==========================================================
for pred_path, gt_path in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
    pred_img = Image.open(pred_path).convert("RGB")
    gt_img   = Image.open(gt_path).convert("RGB")

    if pred_img.size != gt_img.size:
        pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

    pred_np = np.asarray(pred_img, np.float32) / 255.0
    gt_np   = np.asarray(gt_img, np.float32) / 255.0

    psnr_list.append(psnr(gt_np, pred_np, data_range=1.0))
    ssim_list.append(ssim(gt_np, pred_np, channel_axis=2, data_range=1.0))

    with torch.no_grad():
        lp = lpips_fn(to_tensor(pred_img), to_tensor(gt_img)).item()
        lpips_list.append(lp)

# ==========================================================
# 🧩 FID (UNPAIRED — STILL VALID)
# ==========================================================
def load_images(folder):
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, f)).convert("RGB").resize((299, 299))
            arr = np.asarray(img, np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs).to(device)

print("\n🔍 Extracting Inception features for FID...")

x_pred = load_images(pred_dir)
x_gt   = load_images(gt_dir)

try:
    inception = InceptionV3(weights='imagenet', output_blocks=[3]).to(device)
except TypeError:
    inception = InceptionV3(output_blocks=[3]).to(device)

inception.eval()

def inception_feats(x):
    with torch.no_grad():
        f = inception(x)[0]
        return torch.flatten(f, 1)

with torch.no_grad():
    fid_value = fid_metric(
        inception_feats(x_pred),
        inception_feats(x_gt)
    ).item()

# ==========================================================
# 📊 RESULTS
# ==========================================================
print("\n===== Evaluation Results (ORDER-BASED) =====")
print(f"PSNR  : {np.mean(psnr_list):.4f}")
print(f"SSIM  : {np.mean(ssim_list):.4f}")
print(f"LPIPS : {np.mean(lpips_list):.4f}")
print(f"FID   : {fid_value:.4f}")
