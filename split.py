import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


# -----------------------------
# KNet
# -----------------------------
class KNet(nn.Module):
    def __init__(self, in_channels=4, k_max=5):
        super().__init__()

        self.k_max = k_max
        self.head = nn.Conv2d(in_channels, 32, 3, 1, 1)

        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)
        self.res3 = ResidualBlock(32)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1)
        )

        self.tail = nn.Conv2d(32, 1, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, I_max):
        x = torch.cat([rgb, I_max], dim=1)

        x = self.relu(self.head(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        g = self.global_pool(x)
        g = self.global_fc(g)

        x = x + g

        x = self.sigmoid(self.tail(x))
        k_map = 1.0 + x * self.k_max
        k_map = torch.clamp(k_map, 1.0, 6.0)

        return k_map


# -----------------------------
# HVI CONVERSION (IMPORTANT)
# -----------------------------
def rgb_to_hvi(x):
    R, G, B = x[:,0:1], x[:,1:2], x[:,2:3]

    # Intensity
    I = torch.max(x, dim=1, keepdim=True)[0]

    # Improved chroma (more stable than simple subtraction)
    H = R - G
    V = B - 0.5 * (R + G)   # 🔥 FIXED (important)

    return H, V, I


# -----------------------------
# Visualization
# -----------------------------
def normalize(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def save_map(tensor, path, cmap='turbo', power=0.4):
    tensor = tensor.detach().cpu().squeeze().numpy()

    min_val = tensor.min()
    max_val = tensor.max()

    print(f"{path} → min: {min_val:.4f}, max: {max_val:.4f}")

    # 🔥 handle flat maps properly
    if abs(max_val - min_val) < 1e-6:
        print(f"⚠️ {path} is nearly constant → forcing contrast")
        tensor = tensor - min_val
    else:
        tensor = (tensor - min_val) / (max_val - min_val)

    # 🔥 contrast boost (important)
    tensor = np.power(tensor, power)

    plt.imshow(tensor, cmap=cmap)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    image_path = "/home/nitr/Desktop/Mousumi_Behera/Datasets/LOLv1/train/low/537.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = load_image(image_path).to(device)

    # -------------------------
    # HVI decomposition
    # -------------------------
    H, V, I = rgb_to_hvi(x)

    # -------------------------
    # KNet
    # -------------------------
    knet = KNet().to(device)

    checkpoint = torch.load(
        "/home/nitr/Desktop/Mousumi_Behera/Code/HVI-CIDNet-master/weights/train/epoch_360.pth",
        map_location=device
    )

    knet_weights = {
        k.replace("knet.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("knet.")
    }

    knet.load_state_dict(knet_weights)
    knet.eval()

    with torch.no_grad():
        k_map = knet(x, I)

    # -------------------------
    # Save HVI maps
    # -------------------------
    save_map(H[0], "H_map.png", cmap='turbo')
    save_map(V[0], "V_map.png", cmap='turbo', power=0.3)
    save_map(I[0], "I_map.png", cmap='gray', power=1.0)

    # -------------------------
    # Save k-map
    # -------------------------
    save_map(k_map[0], "k_map.png", cmap='turbo', power=0.3)

    # -------------------------
    # Combined figure (LIKE PAPER)
    # -------------------------
    plt.figure(figsize=(8,3))

    plt.subplot(1,3,1)
    plt.title("Ĥ")
    plt.imshow(normalize(H[0]).cpu().squeeze(), cmap='turbo')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Ṽ")
    plt.imshow(normalize(V[0]).cpu().squeeze(), cmap='turbo')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Ĩ")
    plt.imshow(normalize(I[0]).cpu().squeeze(), cmap='gray')
    plt.axis('off')

    plt.savefig("HVI_maps.png", bbox_inches='tight')
    plt.close()

    print("✅ Done!")