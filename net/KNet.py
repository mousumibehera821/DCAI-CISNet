import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Strong KNet
# -----------------------------
class KNet(nn.Module):
    """
    Strong K-Net for Adaptive HVI

    Input:  RGB + I_max  → [B, 4, H, W]
    Output: k_map        → [B, 1, H, W]
    """

    def __init__(self, in_channels=4, k_max=5):
        super().__init__()

        self.k_max = k_max

        # Feature extraction
        self.head = nn.Conv2d(in_channels, 32, 3, 1, 1)

        # Deep feature learning
        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)
        self.res3 = ResidualBlock(32)

        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1)
        )

        # Fusion + output
        self.tail = nn.Conv2d(32, 1, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, I_max):
        """
        rgb   : [B,3,H,W]
        I_max : [B,1,H,W]
        """

        # -------------------------
        # Input fusion
        # -------------------------
        x = torch.cat([rgb, I_max], dim=1)   # [B,4,H,W]

        # -------------------------
        # Feature extraction
        # -------------------------
        x = self.relu(self.head(x))

        # -------------------------
        # Residual learning
        # -------------------------
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # -------------------------
        # Global context
        # -------------------------
        g = self.global_pool(x)              # [B,32,1,1]
        g = self.global_fc(g)                # [B,32,1,1]

        x = x + g                            # broadcast addition

        # -------------------------
        # Output
        # -------------------------
        x = self.sigmoid(self.tail(x))

        # Scale to k range
        k_map = 1.0 + x * self.k_max        # [1, 1 + k_max]

        # Stability clamp (optional but recommended)
        k_map = torch.clamp(k_map, 1.0, 6.0)

        return k_map