"""
Intuitive YOLO26 Multi-Task Implementation

This architecture is rewritten from traditional YOLO config-style arrays into 
standard PyTorch modular design for high readability.

Modules:
1. YOLONanoBackbone: Feature extractor (Yields P3, P4, P5)
2. YOLONeck: FPN/PAN feature fusion (Yields N3, N4, N5)
3. Heads:
    - Detect: Bounding box & class prediction
    - ClsHead: Global image classification 
    - SegHead: Pixel-level segmentation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Base Building Blocks
# ==========================================

def autopad(k: int, p: int | None = None) -> int:
    """Pad to 'same' shape based on kernel size."""
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Standard Convolution -> BatchNorm -> SiLU block."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(nn.Module):
    """C3k block (stacked bottlenecks)."""
    def __init__(self, c: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        self.m = nn.Sequential(*(Bottleneck(c, c, shortcut=shortcut, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class C3k2(C2f):
    """YOLO26 specialized CSP block (C3k2)."""
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, shortcut: bool = True):
        super().__init__(c1, c2, n=n, shortcut=shortcut, e=e)
        self.m = nn.ModuleList(
            (C3k(self.c, n=2, shortcut=shortcut) if c3k else Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0)) 
            for _ in range(n)
        )


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """Lightweight spatial attention."""
    def __init__(self, dim: int, num_heads: int = 4, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.head_dim = dim // self.num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim**-0.5

        hidden = dim + self.key_dim * self.num_heads * 2
        self.qkv = Conv(dim, hidden, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        y = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(y)


class PSABlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        heads = max(c // 64, 1)
        self.attn = Attention(c, num_heads=heads, attn_ratio=0.5)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """YOLO26 C2PSA block (Channel-to-Pixel Spatial Attention)."""
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2, "C2PSA expects c1 == c2"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))


class CrossAttention(nn.Module):
    """Lightweight cross-attention bounded by maximum spatial context to prevent OOM."""
    def __init__(self, qkv_dim: list[int], num_heads: int = 4, max_kv_size: int = 16):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.max_kv_size = max_kv_size

        out_dim = qkv_dim[0]
        self.wq = Conv(qkv_dim[0], out_dim, 1, act=False)
        self.wk = Conv(qkv_dim[1], out_dim, 1, act=False)
        self.wv = Conv(qkv_dim[2], out_dim, 1, act=False)
        self.proj = Conv(out_dim, out_dim, 1, act=False)
        self.pe = Conv(out_dim, out_dim, 3, 1, g=out_dim, act=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        head_dim = c // self.num_heads
        return x.reshape(b, self.num_heads, head_dim, n).permute(0, 1, 3, 2)

    def merge_heads(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, heads, n, head_dim = x.shape
        return x.permute(0, 1, 3, 2).contiguous().view(b, heads * head_dim, h, w)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, _, hq, wq = q.shape
        
        # Bound spatial dimension of K and V to prevent quadratic memory explosion (OOM)
        if k.shape[-1] > self.max_kv_size or k.shape[-2] > self.max_kv_size:
            k = F.adaptive_avg_pool2d(k, (self.max_kv_size, self.max_kv_size))
            v = F.adaptive_avg_pool2d(v, (self.max_kv_size, self.max_kv_size))

        q_split = self.split_heads(self.wq(q))
        k_split = self.split_heads(self.wk(k))
        v_split = self.split_heads(self.wv(v))

        scale = q_split.shape[-1] ** -0.5
        attn = (q_split @ k_split.transpose(-2, -1)) * scale  
        attn = attn.softmax(dim=-1)
        
        y = self.merge_heads(attn @ v_split, hq, wq) + self.pe(self.merge_heads(q_split, hq, wq))
        return self.proj(y)


# ==========================================
# 2. Main Architecture Modules
# ==========================================

class YOLONanoBackbone(nn.Module):
    """
    Feature Extractor (Nano Scale). 
    Processes images and extracts hierarchical spatial features.
    """
    def __init__(self):
        super().__init__()
        # Nano scale channel dimensions
        c1, c2, c3, c4, c5 = 16, 32, 64, 128, 256

        # Stage 1: Stem + Init (Downsamples by 4)
        self.stage1 = nn.Sequential(
            Conv(3, c1, k=3, s=2),
            Conv(c1, c2, k=3, s=2),
            C3k2(c2, c3, n=1, c3k=False, e=0.25, shortcut=False)
        )

        # Stage 2: Downsamples by 8 (Yields P3)
        self.stage2 = nn.Sequential(
            Conv(c3, c3, k=3, s=2),
            C3k2(c3, c4, n=1, c3k=False, e=0.25, shortcut=False)
        )

        # Stage 3: Downsamples by 16 (Yields P4)
        self.stage3 = nn.Sequential(
            Conv(c4, c4, k=3, s=2),
            C3k2(c4, c4, n=1, c3k=True, e=0.5, shortcut=True)
        )

        # Stage 4: Downsamples by 32 (Yields P5)
        self.stage4 = nn.Sequential(
            Conv(c4, c5, k=3, s=2),
            C3k2(c5, c5, n=1, c3k=True, e=0.5, shortcut=True),
            SPPF(c5, c5, k=5, n=3),
            C2PSA(c5, c5, n=1)
        )

        # Expose final channel dims for the Neck to use
        self.channels = {"p3": c4, "p4": c4, "p5": c5}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class YOLONeck(nn.Module):
    """
    Feature Pyramid Network (FPN) & Path Aggregation Network (PAN).
    Fuses features from different backbone stages.
    """
    def __init__(self, in_channels: dict[str, int]):
        super().__init__()
        c3, c4, c5 = 64, 128, 256  # Hardcoded Nano channels (p3 actually outputs 128 channels in backbone)
        # Note: Backbone p3 outputs c4(128). We respect this flow.
        p3_ch, p4_ch, p5_ch = in_channels["p3"], in_channels["p4"], in_channels["p5"]

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Top-Down Pathway (FPN) ---
        self.fpn_p4 = C3k2(p5_ch + p4_ch, c4, n=1, c3k=True, e=0.5, shortcut=True)
        self.n4_attn = CrossAttention(qkv_dim=[c4, p3_ch, p3_ch], num_heads=max(c4 // 32, 1))
        
        self.fpn_p3 = C3k2(c4 + p3_ch, c3, n=1, c3k=True, e=0.5, shortcut=True)

        # --- Bottom-Up Pathway (PAN) ---
        self.down_n3 = Conv(c3, c3, k=3, s=2)
        self.pan_n4 = C3k2(c3 + c4, c4, n=1, c3k=True, e=0.5, shortcut=True)

        self.down_n4 = Conv(c4, c4, k=3, s=2)
        self.pan_n5 = C3k2(c4 + p5_ch, c5, n=1, c3k=True, e=0.5, shortcut=True)
        self.n5_attn = CrossAttention(qkv_dim=[c5, p4_ch, p4_ch], num_heads=max(c5 // 32, 1))

        # Expose final channel dims for the Heads to use
        self.channels = {"n3": c3, "n4": c4, "n5": c5}

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Top-Down (FPN)
        n4 = self.fpn_p4(torch.cat([self.up(p5), p4], dim=1))
        n4 = self.n4_attn(n4, p3, p3)
        n3 = self.fpn_p3(torch.cat([self.up(n4), p3], dim=1))

        # Bottom-Up (PAN)
        n4_out = self.pan_n4(torch.cat([self.down_n3(n3), n4], dim=1))
        n5_out = self.pan_n5(torch.cat([self.down_n4(n4_out), p5], dim=1))
        n5_out = self.n5_attn(n5_out, p4, p4)

        return n3, n4_out, n5_out


# ==========================================
# 3. Heads & Multi-Task Wrapper
# ==========================================

class Detect(nn.Module):
    """Detection Head for Bounding Boxes and Classifications."""
    def __init__(self, nc: int, ch: tuple[int, int, int], reg_max: int = 1):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max

        # Box branch internal channels
        c_box = max(16, ch[0] // 4, self.reg_max * 4)
        # Class branch internal channels
        c_cls = max(ch[0], min(self.nc, 100))

        # Box regression branches
        self.box_branches = nn.ModuleList(
            nn.Sequential(Conv(x, c_box, 3), Conv(c_box, c_box, 3), nn.Conv2d(c_box, 4 * self.reg_max, 1)) 
            for x in ch
        )
        
        # Classification branches
        self.cls_branches = nn.ModuleList(
            nn.Sequential(Conv(x, c_cls, 3), Conv(c_cls, c_cls, 3), nn.Conv2d(c_cls, self.nc, 1)) 
            for x in ch
        )

    def forward(self, neck_features: list[torch.Tensor]):
        bs = neck_features[0].shape[0]
        
        boxes = torch.cat([self.box_branches[i](feat).view(bs, 4 * self.reg_max, -1) for i, feat in enumerate(neck_features)], dim=-1)
        scores = torch.cat([self.cls_branches[i](feat).view(bs, self.nc, -1) for i, feat in enumerate(neck_features)], dim=-1)
        
        return {"boxes": boxes, "scores": scores, "feats": neck_features}


class HandGestureMultiTask(nn.Module):
    """
    Complete Multi-Task Architecture.
    Combines Backbone, Neck, and individual heads (Detection, Classification, Segmentation).
    """
    def __init__(self, num_classes: int = 10, reg_max: int = 1):
        super().__init__()
        
        # 1. Feature Extraction & Fusion
        self.backbone = YOLONanoBackbone()
        self.neck = YOLONeck(in_channels=self.backbone.channels)
        
        # 2. Detection Head (Operates on fused neck features N3, N4, N5)
        neck_ch = (self.neck.channels["n3"], self.neck.channels["n4"], self.neck.channels["n5"])
        self.detect = Detect(nc=num_classes, ch=neck_ch, reg_max=reg_max)

        # 3. Classification Head (Operates on P5 backbone features for rich global semantics)
        c5 = self.backbone.channels["p5"]
        self.cls_head = nn.Sequential(
            Conv(c5, 1280, k=1, s=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        # 4. Segmentation Head (Operates on N3, N4, N5)
        seg_ch = 128
        self.seg_reduce3 = Conv(neck_ch[0], seg_ch, k=1)
        self.seg_reduce4 = Conv(neck_ch[1], seg_ch, k=1)
        self.seg_reduce5 = Conv(neck_ch[2], seg_ch, k=1)

        self.seg_conv = nn.Sequential(
            Conv(seg_ch * 3, 256, 3), 
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            nn.Conv2d(256, 1, kernel_size=1) 
        )

    def freeze_base_model(self):
        """Freezes the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, tasks: tuple[str, ...] | None = None):
        if tasks is None:
            tasks = ("det", "cls", "seg")

        # --- Feature Extraction ---
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)

        out: dict[str, torch.Tensor | object] = {}

        # --- A. Detection Output ---
        if "det" in tasks:
            out["det"] = self.detect([n3, n4, n5])

        # --- B. Classification Output ---
        if "cls" in tasks:
            out["cls"] = self.cls_head(p5)

        # --- C. Segmentation Output ---
        if "seg" in tasks:
            target_size = n3.shape[-2:]
            
            # Reduce channels FIRST to save FLOPs, then upscale
            n3_r = self.seg_reduce3(n3)
            n4_r = F.interpolate(self.seg_reduce4(n4), size=target_size, mode="nearest")
            n5_r = F.interpolate(self.seg_reduce5(n5), size=target_size, mode="nearest")
            
            fused = torch.cat([n3_r, n4_r, n5_r], dim=1)
            mask_low_res = self.seg_conv(fused)
            
            out["seg"] = F.interpolate(mask_low_res, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


# ==========================================
# Execution Demo
# ==========================================
def _demo() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HandGestureMultiTask()
    model.to(device)
    model.eval()

    try:
        import torchinfo
        torchinfo.summary(model, input_size=(1, 3, 256, 256), device=str(device), depth=4)
    except ImportError:
        print("torchinfo not installed. Skipping summary.")

    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        preds = model(x)

    print(f"Detection boxes shape: {preds['det']['boxes'].shape}")
    print(f"Detection scores shape: {preds['det']['scores'].shape}")
    print(f"Detection feats (N3, N4, N5): {[f.shape for f in preds['det']['feats']]}")
    print(f"Classification output shape: {preds['cls'].shape}")
    print(f"Segmentation mask shape: {preds['seg'].shape}")


if __name__ == "__main__":
    _demo()