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
import math

import torchinfo



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


class C3k2(nn.Module):
    """YOLO26 specialized CSP block (C3k2)."""
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            (C3k(self.c, n=2, shortcut=shortcut) if c3k else Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0)) 
            for _ in range(n)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


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



class RGB_V1_BackBone(nn.Module):
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
        )

        # Expose channel dims for downstream heads and the neck.
        self.channels = {"p2": c3, "p3": c4, "p4": c4, "p5": c5}

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        return p5


class DetHead_V1(nn.Module):
    """Predict exactly one bounding box and one class per image."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        hidden_channels = max(in_channels // 2, 32)

        self.stem = nn.Sequential(
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, hidden_channels, 3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bbox_head = nn.Linear(hidden_channels, 4)
        self.class_head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        y = self.pool(x).flatten(1)

        bbox = self.bbox_head(y).unsqueeze(-1)
        class_logits = self.class_head(y).unsqueeze(-1)
        return {"boxes": bbox, "scores": class_logits}


class ClsHead(nn.Module):
    """Predict a single image-level class from the deepest backbone feature map."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        hidden_channels = max(in_channels // 2, 64)

        self.cls_head = nn.Sequential(
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, hidden_channels, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(x)


class SegHead(nn.Module):
    """Binary segmentation head that predicts masks using only the p5 feature map."""

    def __init__(self, in_channels: int, seg_ch: int = 64):
        super().__init__()
        self.reduce = Conv(in_channels, seg_ch * 2, k=1, s=1)
        self.seg_conv = nn.Sequential(
            Conv(seg_ch * 2, seg_ch * 2, 3),
            Conv(seg_ch * 2, seg_ch, 3),
            nn.Conv2d(seg_ch, 1, kernel_size=1),
        )

    def forward(
        self,
        p5: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        mask_logits = self.seg_conv(self.reduce(p5))

        if output_size is None:
            output_size = p5.shape[-2:]

        return F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)


class DetHead_V2(nn.Module):
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

        # Initialize the final classification Conv layer bias to predict ~1% probability initially
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for branch in self.cls_branches:
            # branch[-1] is the final nn.Conv2d(c_cls, self.nc, 1)
            nn.init.constant_(branch[-1].bias, bias_value)


    def forward(self, neck_features: list[torch.Tensor]):
        bs = neck_features[0].shape[0]
        
        boxes = torch.cat([self.box_branches[i](feat).view(bs, 4 * self.reg_max, -1) for i, feat in enumerate(neck_features)], dim=-1)
        scores = torch.cat([self.cls_branches[i](feat).view(bs, self.nc, -1) for i, feat in enumerate(neck_features)], dim=-1)
        
        return {"boxes": boxes, "scores": scores, "feats": neck_features}


class RGB_V1(nn.Module):
    def __init__(self, num_classes: int = 10, reg_max = None):
        super().__init__()
        self.backbone = RGB_V1_BackBone()
        self.detect = DetHead_V1(in_channels=self.backbone.channels["p5"], num_classes=10)
        self.cls_head = ClsHead(in_channels=self.backbone.channels["p5"], num_classes=10)
        self.seg_head = SegHead(in_channels=self.backbone.channels["p5"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p5 = self.backbone(x)

        det_outputs = self.detect(p5)
        cls_logits = self.cls_head(p5)
        seg_logits = self.seg_head(p5, output_size=x.shape[-2:])

        return {
            "det": det_outputs,
            "cls": cls_logits,
            "seg": seg_logits
        }


class RGB_V2(nn.Module):
    def __init__(self, num_classes: int = 10, reg_max = None):
        super().__init__()
        self.backbone = RGB_V1_BackBone()
        self.detect = DetHead_V2(nc=10, ch=(256,), reg_max=1)
        self.cls_head = ClsHead(in_channels=self.backbone.channels["p5"], num_classes=10)
        self.seg_head = SegHead(in_channels=self.backbone.channels["p5"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p5 = self.backbone(x)

        det_outputs = self.detect([p5])
        cls_logits = self.cls_head(p5)
        seg_logits = self.seg_head(p5, output_size=x.shape[-2:])

        return {
            "det": det_outputs,
            "cls": cls_logits,
            "seg": seg_logits
        }






if __name__ == "__main__":

    print("Testing RGB_V1 Backbone and Heads with dummy input...")

    print(f"v1 backbone:")
    x = torch.randn(1, 3, 480, 640)
    v1_backbone = RGB_V1_BackBone()
    p5 = v1_backbone(x)
    print(p5.shape)

    print(f"detection head v1:")
    det_head_v1 = DetHead_V1(in_channels=256, num_classes=10)
    det_outputs = det_head_v1(p5)
    print(det_outputs["boxes"].shape)
    print(det_outputs["scores"].shape)

    print(f"detection head v2:")
    det_head_v2 = DetHead_V2(nc=10, ch=(256,), reg_max=1)
    det_outputs_v2 = det_head_v2([p5])
    print(det_outputs_v2["boxes"].shape)
    print(det_outputs_v2["scores"].shape)
    print(det_outputs_v2["feats"][0].shape)

    print(f"classification head:")
    cls_head = ClsHead(in_channels=256, num_classes=10)
    cls_logits = cls_head(p5)
    print(cls_logits.shape)

    print(f"segmentation head:")
    seg_head = SegHead(in_channels=256)
    seg_logits = seg_head(p5, output_size=x.shape[-2:])
    print(seg_logits.shape)

    print("\nTesting RGB_V1 full model:")
    v1_model = RGB_V1()
    torchinfo.summary()
    outputs = v1_model(x)

    print("\nRGB_V2 full model test:")
    v2_model = RGB_V2(num_classes=10, reg_max=1)
    outputs = v2_model(x)
    print(outputs["det"]["boxes"].shape)
    print(outputs["det"]["scores"].shape)
    print(outputs["det"]["feats"][0].shape)
    print(outputs["cls"].shape)
    print(outputs["seg"].shape)
    