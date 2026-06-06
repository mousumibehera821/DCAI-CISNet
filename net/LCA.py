import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------
# Depthwise Separable Convolution
# -----------------------------
class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


# -----------------------------
# Cross Attention Block (CAB)  ← SAME NAME
# -----------------------------
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias=False, dropout=0.0):
        super(CAB, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # UPDATED: Depthwise-based Q, K, V
        self.q = Depth_conv(dim, dim)
        self.k = Depth_conv(dim, dim)
        self.v = Depth_conv(dim, dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # Q, K, V
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        # Multi-head reshape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize (important!)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v

        # Restore spatial
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


# -----------------------------
# Intensity Enhancement Layer (UNCHANGED)
# -----------------------------
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, padding=1, groups=hidden_features * 2)

        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features,
                                 kernel_size=3, padding=1, groups=hidden_features)

        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features,
                                 kernel_size=3, padding=1, groups=hidden_features)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.act = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x1 = self.act(self.dwconv1(x1)) + x1
        x2 = self.act(self.dwconv2(x2)) + x2

        x = x1 * x2
        x = self.project_out(x)

        return x


# -----------------------------
# LayerNorm (UNCHANGED)
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


# -----------------------------
# HV_LCA (SAME NAME, improved CAB inside)
# -----------------------------
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


# -----------------------------
# I_LCA (SAME NAME, improved CAB inside)
# -----------------------------
class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x