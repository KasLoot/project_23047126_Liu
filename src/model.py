"""
Standalone YOLO26 implementation (from-scratch style) for architecture inspection.

This file intentionally re-implements the key YOLO26 detect architecture components
without importing Ultralytics internal modules, so the design is easier to inspect
and modify.

Covered features:
- Compound scaling (n/s/m/l/x)
- Backbone: Conv -> C3k2 blocks -> SPPF -> C2PSA
- PAN/FPN neck with P3/P4/P5 outputs
- Dual-head detection design (one-to-many + one-to-one branches)
- End-to-end path helper (top-k selection, NMS-free style output format)

Notes:
- This is an educational/reference implementation, not weight-compatible with
  official Ultralytics checkpoints.
- It mimics the YOLO26 detect YAML topology and high-level head behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------


def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Conv-BN-Activation block."""

    default_act = nn.SiLU

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

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
    """C2f-style CSP block."""

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
    """C3k block (stacked bottlenecks with configurable kernel-like behavior)."""

    def __init__(self, c: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        self.m = nn.Sequential(*(Bottleneck(c, c, shortcut=shortcut, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class C3k2(C2f):
    """YOLO26 C3k2 block.

    For simplicity, this implementation uses the C2f scaffold with each repeated block
    being either Bottleneck or C3k.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n=n, shortcut=shortcut, e=e)
        self.m = nn.ModuleList((C3k(self.c, n=2, shortcut=shortcut) if c3k else Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0)) for _ in range(n))


class SPPF(nn.Module):
    """Fast SPP block."""

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
    """Lightweight spatial attention used by PSA blocks."""

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
    """YOLO26 C2PSA block."""

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


class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=self.dim)


@dataclass
class ScaleCfg:
    depth: float
    width: float
    max_channels: int


SCALES: dict[str, ScaleCfg] = {
    "n": ScaleCfg(0.50, 0.25, 1024),
    "s": ScaleCfg(0.50, 0.50, 1024),
    "m": ScaleCfg(0.50, 1.00, 512),
    "l": ScaleCfg(1.00, 1.00, 512),
    "x": ScaleCfg(1.00, 1.50, 512),
}


def make_divisible(v: float, divisor: int = 8) -> int:
    return int((v + divisor / 2) // divisor * divisor)


def scale_channels(c: int, w: float, max_ch: int) -> int:
    return make_divisible(min(c, max_ch) * w, 8)


def scale_depth(n: int, d: float) -> int:
    return max(round(n * d), 1) if n > 1 else n


class CrossAttention(nn.Module):
    """Lightweight spatial attention used by PSA blocks."""

    def __init__(self, qkv_dim: list[int], num_heads: int = 4, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(num_heads, 1)

        out_dim = qkv_dim[0]
        self.wq = Conv(qkv_dim[0], out_dim, 1, act=False)
        self.wk = Conv(qkv_dim[1], out_dim, 1, act=False)
        self.wv = Conv(qkv_dim[2], out_dim, 1, act=False)
        self.proj = Conv(out_dim, out_dim, 1, act=False)
        self.pe = Conv(out_dim, out_dim, 3, 1, g=out_dim, act=False)

        if out_dim % self.num_heads != 0:
            raise ValueError(f"CrossAttention out_dim={out_dim} must be divisible by num_heads={self.num_heads}")

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        head_dim = c // self.num_heads
        # (B, C, H, W) -> (B, heads, N, head_dim)
        return x.reshape(b, self.num_heads, head_dim, n).permute(0, 1, 3, 2)

    def merge_heads(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # (B, heads, N, head_dim) -> (B, C, H, W)
        b, heads, n, head_dim = x.shape
        return x.permute(0, 1, 3, 2).contiguous().view(b, heads * head_dim, h, w)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, _, hq, wq = q.shape
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)  # (B, heads, Nq, d)
        k = self.split_heads(k)  # (B, heads, Nk, d)
        v = self.split_heads(v)  # (B, heads, Nk, d)

        scale = q.shape[-1] ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, Nq, Nk)
        attn = attn.softmax(dim=-1)
        y = attn @ v  # (B, heads, Nq, d)
        y = self.merge_heads(y, hq, wq) + self.pe(self.merge_heads(q, hq, wq))
        return self.proj(y)


class BackBone(nn.Module):
    """YOLO26 detect model (from scratch / inspectable implementation)."""

    def __init__(self, nc: int = 10, scale: Literal["n", "s", "m", "l", "x"] = "n", end2end: bool = True, reg_max: int = 1):
        super().__init__()
        if scale not in SCALES:
            raise ValueError(f"Unsupported scale '{scale}'. Choose from {list(SCALES)}")

        cfg = SCALES[scale]
        d, w, max_ch = cfg.depth, cfg.width, cfg.max_channels

        # --- Backbone channels ---
        self.c1 = scale_channels(64, w, max_ch)
        self.c2 = scale_channels(128, w, max_ch)
        self.c3 = scale_channels(256, w, max_ch)
        self.c4 = scale_channels(512, w, max_ch)
        self.c5 = scale_channels(1024, w, max_ch)
        print(f"Scaled channels (c1-c5): {self.c1}, {self.c2}, {self.c3}, {self.c4}, {self.c5}")

        n2 = scale_depth(2, d)

        # Backbone
        self.b0 = Conv(3, self.c1, 3, 2)  # P1/2
        self.b1 = Conv(self.c1, self.c2, 3, 2)  # P2/4
        self.b2 = C3k2(self.c2, self.c3, n=n2, c3k=False, e=0.25, shortcut=False)

        self.b3 = Conv(self.c3, self.c3, 3, 2)  # P3/8
        self.b4 = C3k2(self.c3, self.c4, n=n2, c3k=False, e=0.25, shortcut=False)

        self.b5 = Conv(self.c4, self.c4, 3, 2)  # P4/16
        self.b6 = C3k2(self.c4, self.c4, n=n2, c3k=True, e=0.5, shortcut=True)

        self.b7 = Conv(self.c4, self.c5, 3, 2)  # P5/32
        self.b8 = C3k2(self.c5, self.c5, n=n2, c3k=True, e=0.5, shortcut=True)

        self.b9 = SPPF(self.c5, self.c5, k=5, n=3)
        self.b10 = C2PSA(self.c5, self.c5, n=n2)

        # Neck / head feature aggregation
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cat = Concat(1)

        self.h13 = C3k2(self.c5 + self.c4, self.c4, n=n2, c3k=True, e=0.5, shortcut=True)
        self.h16 = C3k2(self.c4 + self.c4, self.c3, n=n2, c3k=True, e=0.5, shortcut=True)

        self.h17 = Conv(self.c3, self.c3, 3, 2)
        self.h19 = C3k2(self.c3 + self.c4, self.c4, n=n2, c3k=True, e=0.5, shortcut=True)

        self.h20 = Conv(self.c4, self.c4, 3, 2)
        self.h22 = C3k2(self.c4 + self.c5, self.c5, n=1, c3k=True, e=0.5, shortcut=True)

        # NOTE: In this backbone, p3 is produced by `b4(self.b3(...))` and has `c4` channels.
        # Keep q/k/v channel dims consistent with the actual tensors passed in forward().
        self.n4_attn = CrossAttention(qkv_dim=[self.c4, self.c4, self.c4], num_heads=max(self.c4 // 32, 1), attn_ratio=0.5)
        self.n5_attn = CrossAttention(qkv_dim=[self.c5, self.c4, self.c4], num_heads=max(self.c5 // 32, 1), attn_ratio=0.5)

        self.nc = nc
        self.scale = scale
        self.end2end = end2end
        self.reg_max = reg_max

    def forward(self, x: torch.Tensor):
        # Backbone
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)

        p3 = self.b3(x)
        p3 = self.b4(p3)

        p4 = self.b5(p3)
        p4 = self.b6(p4)

        p5 = self.b7(p4)
        p5 = self.b8(p5)
        p5 = self.b9(p5)
        p5 = self.b10(p5)

        # PAN/FPN neck
        n4 = self.h13(self.cat([self.up(p5), p4]))  # P4/16
        n4 = self.n4_attn(n4, p3, p3)
        n3 = self.h16(self.cat([self.up(n4), p3]))  # P3/8 (small)

        n4_out = self.h19(self.cat([self.h17(n3), n4]))  # P4/16 (medium)
        n5_out = self.h22(self.cat([self.h20(n4_out), p5]))  # P5/32 (large)
        n5_out = self.n5_attn(n5_out, p4, p4)

        return [n3, n4_out, n5_out]




# -----------------------------
# Detection head (simplified)
# -----------------------------


class Detect(nn.Module):
    """YOLO26-style dual-branch head.

    - one-to-many: conventional dense supervision branch
    - one-to-one: detached-feature branch for end-to-end NMS-free inference

    This variant keeps `reg_max=1` by default (DFL removed behavior), matching YOLO26
    config where regression outputs are direct 4-dim distances.
    """

    def __init__(self, nc: int = 10, ch: tuple[int, int, int] = (256, 512, 1024), reg_max: int = 1, end2end: bool = True):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max
        self.end2end = end2end
        self.max_det = 300

        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

        if end2end:
            import copy

            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def _forward_branch(self, x: list[torch.Tensor], box_head: nn.ModuleList, cls_head: nn.ModuleList) -> dict[str, torch.Tensor]:
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def _postprocess_one2one(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # boxes: (B, 4, A), scores: (B, nc, A)
        b, _, a = boxes.shape
        boxes = boxes.permute(0, 2, 1)  # (B, A, 4)
        scores = scores.sigmoid().permute(0, 2, 1)  # (B, A, nc)

        conf, cls_idx = scores.max(dim=-1, keepdim=True)  # (B, A, 1)
        k = min(self.max_det, a)
        topk_conf, topk_idx = conf.squeeze(-1).topk(k, dim=1)

        gather4 = topk_idx.unsqueeze(-1).expand(b, k, 4)
        gather1 = topk_idx.unsqueeze(-1)
        topk_boxes = boxes.gather(1, gather4)
        topk_cls = cls_idx.gather(1, gather1).float()
        topk_conf = topk_conf.unsqueeze(-1)

        # (B, K, 6): [x, y, w, h, conf, cls]
        return torch.cat([topk_boxes, topk_conf, topk_cls], dim=-1)

    def forward(self, x: list[torch.Tensor]):
        one2many = self._forward_branch(x, self.cv2, self.cv3)

        return one2many






class HandGestureMultiTask(nn.Module):
    def __init__(
        self,
        yolo_weights_path: str | None = None,
        num_classes: int = 10,
        end2end: bool = True,
        scale: Literal["n", "s", "m", "l", "x"] = "m",
        reg_max: int = 1,
    ):
        super().__init__()
        
        # 1. Load the base YOLO26 model
        self.backbone = BackBone(nc=num_classes, scale=scale, end2end=end2end, reg_max=reg_max)
        

        detec_ch = (self.backbone.c3, self.backbone.c4, self.backbone.c5)
        self.detect = Detect(nc=num_classes, ch=detec_ch, reg_max=reg_max, end2end=end2end)

        # 2. Extract channel sizes dynamically from the pre-built Detect head
        # This ensures it always matches your backbone scale (n, s, m, l, x)
        c3 = self.detect.cv2[0][0].conv.in_channels
        c4 = self.detect.cv2[1][0].conv.in_channels
        c5 = self.detect.cv2[2][0].conv.in_channels

        # 3. Classification Head (Global image context)
        self.cls_head = nn.Sequential(
            # 1. Process the spatial layout while it still exists
            Conv(c5, 256, k=3, s=1),
            Conv(256, 256, k=3, s=2), # Stride 2 to downsample spatially
            Conv(256, 256, k=3, s=1),
            
            # 2. Now that we've learned complex spatial patterns, squash it
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 3. Deeper fully connected network with stronger dropout
            nn.Dropout(0.3), # Increased dropout to prevent overfitting this large head
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), # Batch norm helps deep linear networks
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 4. Segmentation Head (Pixel-level context)
        # self.seg_conv = nn.Sequential(
        #     Conv(c3 + c4 + c5, 256, 3), # Fuse the three neck scales
        #     Conv(256, 128, 3),
        #     nn.Conv2d(128, 1, kernel_size=1) # Output 1 channel for binary hand mask
        # )
        self.seg_conv = nn.Sequential(
            Conv(c3 + c4 + c5, 256, 3), # Fuse the three neck scales
            Conv(256, 256, k=3, s=2), # Stride 2 to downsample spatially
            Conv(256, 256, k=3, s=1),
            Attention(256, num_heads=4, attn_ratio=0.5), # Add attention for better spatial understanding
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            nn.Conv2d(256, 1, kernel_size=1) # Output 1 channel for binary hand mask
        )

        self.seg_conv = nn.ModuleList([
            Conv(c3 + c4 + c5, 256, 3), # Fuse the three neck scales
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            # Attention(256, num_heads=4, attn_ratio=0.5), # Add attention for better spatial understanding
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            nn.Conv2d(256, 1, kernel_size=1) # Output 1 channel for binary hand mask
        ])



    def freeze_base_model(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen.")

    def forward(self, x: torch.Tensor, tasks: tuple[str, ...] | None = None):
        # Default keeps backwards-compat: compute all heads.
        if tasks is None:
            tasks = ("det", "cls", "seg")
        # # --- 1. Run Backbone ---
        # b_x = self.backbone.b0(x)
        # b_x = self.backbone.b1(b_x)
        # b_x = self.backbone.b2(b_x)
        # p3 = self.backbone.b4(self.backbone.b3(b_x))
        # p4 = self.backbone.b6(self.backbone.b5(p3))
        # p5 = self.backbone.b10(self.backbone.b9(self.backbone.b8(self.backbone.b7(p4))))

        # # --- 2. Run Neck ---
        # n4 = self.backbone.h13(self.backbone.cat([self.backbone.up(p5), p4]))
        # n3 = self.backbone.h16(self.backbone.cat([self.backbone.up(n4), p3]))
        # n4_out = self.backbone.h19(self.backbone.cat([self.backbone.h17(n3), n4]))
        # n5_out = self.backbone.h22(self.backbone.cat([self.backbone.h20(n4_out), p5]))

        neck_features = self.backbone(x)
        n3, n4, n5 = neck_features[0], neck_features[1], neck_features[2]

        out: dict[str, torch.Tensor | object] = {}

        # A. Detection Output
        if "det" in tasks:
            out["det"] = self.detect(neck_features)

        # B. Classification Output
        if "cls" in tasks:
            out["cls"] = self.cls_head(n5)

        # C. Segmentation Output
        if "seg" in tasks:
            target_size = n3.shape[-2:]
            n4_up = F.interpolate(n4, size=target_size, mode="nearest")
            n5_up = F.interpolate(n5, size=target_size, mode="nearest")
            fused = self.backbone.cat([n3, n4_up, n5_up])

            for layer in self.seg_conv:
                if isinstance(layer, Conv) or isinstance(layer, nn.Conv2d):
                    fused = layer(fused)
                if isinstance(layer, Attention):
                    fused = layer(fused) + fused
            mask_low_res = fused
            out["seg"] = F.interpolate(mask_low_res, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out










# -----------------------------
# Architecture summary helpers
# -----------------------------
def _demo() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HandGestureMultiTask()
    model.to(device)
    model.eval()

    import torchinfo
    torchinfo.summary(model, input_size=(1, 3, 256, 256), device=str(device), depth=7)


    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        preds = model(x)

    det_preds = preds["det"]
    cls_preds = preds["cls"]
    seg_preds = preds["seg"]
    print(f"Detection output boxes shape: {det_preds['boxes'].shape}")
    print(f"Detection output scores shape: {det_preds['scores'].shape}")
    print(f"Detecton output feats (neck features) lengths: {[f.shape for f in det_preds['feats']]}")
    print(f"Classification output shape: {cls_preds.shape}")
    print(f"Segmentation output shape: {seg_preds.shape}")

    print("\n\n\n")
    print("Test Cross-Attention module with dummy inputs:")
    attn = CrossAttention(qkv_dim=[128, 64, 64], num_heads=4, attn_ratio=0.5).to(device)
    q = torch.randn(1, 128, 16, 16, device=device)
    k = torch.randn(1, 64, 32, 32, device=device)
    v = torch.randn(1, 64, 32, 32, device=device)
    attn_out = attn(q, k, v)
    print(f"Cross-Attention output shape: {attn_out.shape}")



if __name__ == "__main__":
    _demo()
