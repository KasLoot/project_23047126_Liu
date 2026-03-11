"""
Intuitive YOLO-style Multi-Task Implementation

Modules:
1. RGB_Base: Clean, purely convolutional baseline (CSP-based).
2. RGB_Dynamic: Uses Selective Kernel (SK) multi-branch dynamic convolutions.
3. RGB_Attention: Uses Coordinate Attention (CoordAtt) and Cross-Scale Gating.
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

class C3k(nn.Module):
    def __init__(self, c: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        self.m = nn.Sequential(*(Bottleneck(c, c, shortcut=shortcut, e=1.0) for _ in range(n)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

class C3k2(nn.Module):
    """CSP block."""
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


# ==========================================
# 2. Advanced Dynamic & Attention Blocks
# ==========================================

class SKConv(nn.Module):
    """Selective Kernel Convolution (Dynamic Multi-Branch Convolution)."""
    def __init__(self, features, reduction=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / reduction), L)
        
        # Branch 1: 3x3 receptive field
        self.conv1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.SiLU()
        )
        # Branch 2: 5x5 receptive field (simulated with dilation=2 to save parameters)
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(features),
            nn.SiLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.SiLU()
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(d, features, kernel_size=1, bias=False),
            nn.Conv2d(d, features, kernel_size=1, bias=False)
        ])
        
    def forward(self, x):
        U1 = self.conv1(x)
        U2 = self.conv2(x)
        U = U1 + U2
        
        S = self.gap(U)
        Z = self.fc(S)
        
        A1 = self.fcs[0](Z)
        A2 = self.fcs[1](Z)
        A = torch.cat([A1, A2], dim=1)
        A = F.softmax(A.view(A.shape[0], 2, A.shape[1]//2, 1, 1), dim=1)
        
        V = U1 * A[:, 0] + U2 * A[:, 1]
        return V

class SKBottleneck(nn.Module):
    """Bottleneck using Selective Kernel Convolutions."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.sk_conv = SKConv(c_, stride=1)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.sk_conv(self.cv1(x)))
        return x + y if self.add else y

class C3k2_SK(nn.Module):
    """CSP block utilizing SK (Dynamic) Bottlenecks."""
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(SKBottleneck(self.c, self.c, shortcut=shortcut, e=1.0) for _ in range(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CoordAtt(nn.Module):
    """Coordinate Attention Block."""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

class SemanticGatedFusion(nn.Module):
    """Fuses high-res (shallow) and low-res (deep) features via spatial gating."""
    def __init__(self, high_res_ch, low_res_ch, out_ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            Conv(low_res_ch, high_res_ch, k=1, s=1),
            nn.Conv2d(high_res_ch, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.fusion_conv = Conv(high_res_ch + low_res_ch, out_ch, k=3, s=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, high_res_feat, low_res_feat):
        # low_res_feat is deeply semantic but spatially small
        low_res_up = self.upsample(low_res_feat)
        
        # Calculate spatial attention map from semantic feature
        spatial_gate = self.gate_conv(low_res_up)
        
        # Gate the noisy high-res feature
        gated_high_res = high_res_feat * spatial_gate
        
        # Concat and fuse
        fused = torch.cat([gated_high_res, low_res_up], dim=1)
        return self.fusion_conv(fused)

# ==========================================
# 3. Model 1: RGB_Base (Pure Convolutional)
# ==========================================

class RGB_Base_BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5 = 16, 32, 64, 128, 256
        self.stage1 = nn.Sequential(Conv(3, c1, 3, 2), Conv(c1, c2, 3, 2), C3k2(c2, c3, 1, False, 0.25, False))
        self.stage2 = nn.Sequential(Conv(c3, c3, 3, 2), C3k2(c3, c4, 1, False, 0.25, False))
        self.stage3 = nn.Sequential(Conv(c4, c4, 3, 2), C3k2(c4, c4, 1, True, 0.5, True))
        self.stage4 = nn.Sequential(Conv(c4, c5, 3, 2), C3k2(c5, c5, 1, True, 0.5, True), SPPF(c5, c5, 5, 3))
        self.channels = {"p2": c3, "p3": c4, "p4": c4, "p5": c5}

    def forward(self, x: torch.Tensor, return_p2: bool = False):
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return (p2, p3, p4, p5) if return_p2 else (p3, p4, p5)

class RGB_Base_Neck(nn.Module):
    def __init__(self, in_channels: dict[str, int]):
        super().__init__()
        p3_ch, p4_ch, p5_ch = in_channels["p3"], in_channels["p4"], in_channels["p5"]
        self.c3, self.c4, self.c5 = 64, 128, 256

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # Top-down FPN
        self.fpn_p4 = C3k2(p4_ch + p5_ch, p4_ch, n=1, shortcut=False)
        self.fpn_p3 = C3k2(p3_ch + p4_ch, self.c3, n=1, shortcut=False)
        # Bottom-up PAN
        self.down_p3 = Conv(self.c3, self.c3, k=3, s=2)
        self.pan_p4 = C3k2(self.c3 + p4_ch, self.c4, n=1, shortcut=False)
        self.down_p4 = Conv(self.c4, self.c4, k=3, s=2)
        self.pan_p5 = C3k2(self.c4 + p5_ch, self.c5, n=1, shortcut=False)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        f_p4 = self.fpn_p4(torch.cat([p4, self.up(p5)], dim=1))
        n3 = self.fpn_p3(torch.cat([p3, self.up(f_p4)], dim=1))
        
        n4 = self.pan_p4(torch.cat([self.down_p3(n3), f_p4], dim=1))
        n5 = self.pan_p5(torch.cat([self.down_p4(n4), p5], dim=1))
        return n3, n4, n5

class RGB_Base(nn.Module):
    """Model 1: Highly optimized pure convolutional baseline."""
    def __init__(self, num_classes: int, reg_max: int = 1):
        super().__init__()
        self.backbone = RGB_Base_BackBone()
        self.neck = RGB_Base_Neck(self.backbone.channels)
        neck_ch = (self.neck.c3, self.neck.c4, self.neck.c5)
        self.detect = DetectionHead(nc=num_classes, ch=neck_ch, reg_max=reg_max)
        self.cls_head = ClsHead(nc=num_classes, in_ch=neck_ch[2])
        self.seg_head = SegHead(p2_ch=self.backbone.channels["p2"], neck_ch=neck_ch)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p2, p3, p4, p5 = self.backbone(x, return_p2=True)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return {
            "det": self.detect([n3, n4, n5]),
            "cls": self.cls_head(n5),
            "seg": self.seg_head(p2, [n3, n4, n5], output_size=x.shape[-2:])
        }

# ==========================================
# 4. Model 2: RGB_Dynamic (Multi-Branch/SK)
# ==========================================

class RGB_Dynamic_BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5 = 16, 32, 64, 128, 256
        self.stage1 = nn.Sequential(Conv(3, c1, 3, 2), Conv(c1, c2, 3, 2), C3k2(c2, c3, 1, False, 0.25, False))
        self.stage2 = nn.Sequential(Conv(c3, c3, 3, 2), C3k2(c3, c4, 1, False, 0.25, False))
        # Use Dynamic SK Blocks for deeper stages to handle scale variation
        self.stage3 = nn.Sequential(Conv(c4, c4, 3, 2), C3k2_SK(c4, c4, n=1, e=0.5, shortcut=True))
        self.stage4 = nn.Sequential(Conv(c4, c5, 3, 2), C3k2_SK(c5, c5, n=1, e=0.5, shortcut=True), SPPF(c5, c5, 5, 3))
        self.channels = {"p2": c3, "p3": c4, "p4": c4, "p5": c5}

    def forward(self, x: torch.Tensor, return_p2: bool = False):
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return (p2, p3, p4, p5) if return_p2 else (p3, p4, p5)

class RGB_Dynamic_Neck(nn.Module):
    def __init__(self, in_channels: dict[str, int]):
        super().__init__()
        p3_ch, p4_ch, p5_ch = in_channels["p3"], in_channels["p4"], in_channels["p5"]
        self.c3, self.c4, self.c5 = 64, 128, 256
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        
        # FPN with Dynamic SK blocks
        self.fpn_p4 = C3k2_SK(p4_ch + p5_ch, p4_ch, n=1, shortcut=False)
        self.fpn_p3 = C3k2_SK(p3_ch + p4_ch, self.c3, n=1, shortcut=False)
        
        # PAN with Dynamic SK blocks
        self.down_p3 = Conv(self.c3, self.c3, k=3, s=2)
        self.pan_p4 = C3k2_SK(self.c3 + p4_ch, self.c4, n=1, shortcut=False)
        self.down_p4 = Conv(self.c4, self.c4, k=3, s=2)
        self.pan_p5 = C3k2_SK(self.c4 + p5_ch, self.c5, n=1, shortcut=False)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        f_p4 = self.fpn_p4(torch.cat([p4, self.up(p5)], dim=1))
        n3 = self.fpn_p3(torch.cat([p3, self.up(f_p4)], dim=1))
        n4 = self.pan_p4(torch.cat([self.down_p3(n3), f_p4], dim=1))
        n5 = self.pan_p5(torch.cat([self.down_p4(n4), p5], dim=1))
        return n3, n4, n5

class RGB_Dynamic(nn.Module):
    """Model 2: Dynamic Multi-Branch Convolution (Selective Kernel) to handle scale variation."""
    def __init__(self, num_classes: int, reg_max: int = 1):
        super().__init__()
        self.backbone = RGB_Dynamic_BackBone()
        self.neck = RGB_Dynamic_Neck(self.backbone.channels)
        neck_ch = (self.neck.c3, self.neck.c4, self.neck.c5)
        self.detect = DetectionHead(nc=num_classes, ch=neck_ch, reg_max=reg_max)
        self.cls_head = ClsHead(nc=num_classes, in_ch=neck_ch[2])
        self.seg_head = SegHead(p2_ch=self.backbone.channels["p2"], neck_ch=neck_ch)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p2, p3, p4, p5 = self.backbone(x, return_p2=True)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return {
            "det": self.detect([n3, n4, n5]),
            "cls": self.cls_head(n5),
            "seg": self.seg_head(p2, [n3, n4, n5], output_size=x.shape[-2:])
        }

# ==========================================
# 5. Model 3: RGB_Attention (CoordAtt + Gated Fusion)
# ==========================================

class RGB_Attention_BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3, c4, c5 = 16, 32, 64, 128, 256
        self.stage1 = nn.Sequential(Conv(3, c1, 3, 2), Conv(c1, c2, 3, 2), C3k2(c2, c3, 1, False, 0.25, False))
        
        # Add Coordinate Attention after CSP blocks to retain positional dependencies
        self.stage2 = nn.Sequential(Conv(c3, c3, 3, 2), C3k2(c3, c4, 1, False, 0.25, False), CoordAtt(c4, c4))
        self.stage3 = nn.Sequential(Conv(c4, c4, 3, 2), C3k2(c4, c4, 1, True, 0.5, True), CoordAtt(c4, c4))
        self.stage4 = nn.Sequential(Conv(c4, c5, 3, 2), C3k2(c5, c5, 1, True, 0.5, True), SPPF(c5, c5, 5, 3), CoordAtt(c5, c5))
        
        self.channels = {"p2": c3, "p3": c4, "p4": c4, "p5": c5}

    def forward(self, x: torch.Tensor, return_p2: bool = False):
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return (p2, p3, p4, p5) if return_p2 else (p3, p4, p5)

class RGB_Attention_Neck(nn.Module):
    def __init__(self, in_channels: dict[str, int]):
        super().__init__()
        p3_ch, p4_ch, p5_ch = in_channels["p3"], in_channels["p4"], in_channels["p5"]
        self.c3, self.c4, self.c5 = 64, 128, 256
        
        # Use Semantic Gated Fusion instead of simple concat
        self.fpn_p4_gate = SemanticGatedFusion(p4_ch, p5_ch, p4_ch)
        self.fpn_p3_gate = SemanticGatedFusion(p3_ch, p4_ch, self.c3)
        
        self.down_p3 = Conv(self.c3, self.c3, k=3, s=2)
        self.pan_p4 = C3k2(self.c3 + p4_ch, self.c4, n=1, shortcut=False)
        self.down_p4 = Conv(self.c4, self.c4, k=3, s=2)
        self.pan_p5 = C3k2(self.c4 + p5_ch, self.c5, n=1, shortcut=False)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Top-down gating: deep semantic features dictate which high-res spatial features to keep
        f_p4 = self.fpn_p4_gate(p4, p5)
        n3 = self.fpn_p3_gate(p3, f_p4)
        
        # Bottom-up standard PAN
        n4 = self.pan_p4(torch.cat([self.down_p3(n3), f_p4], dim=1))
        n5 = self.pan_p5(torch.cat([self.down_p4(n4), p5], dim=1))
        return n3, n4, n5

class RGB_Attention(nn.Module):
    """Model 3: Coordinate Attention backbone with Semantic-Gated Cross-Scale Neck."""
    def __init__(self, num_classes: int, reg_max: int = 1):
        super().__init__()
        self.backbone = RGB_Attention_BackBone()
        self.neck = RGB_Attention_Neck(self.backbone.channels)
        neck_ch = (self.neck.c3, self.neck.c4, self.neck.c5)
        self.detect = DetectionHead(nc=num_classes, ch=neck_ch, reg_max=reg_max)
        self.cls_head = ClsHead(nc=num_classes, in_ch=neck_ch[2])
        self.seg_head = SegHead(p2_ch=self.backbone.channels["p2"], neck_ch=neck_ch)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p2, p3, p4, p5 = self.backbone(x, return_p2=True)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return {
            "det": self.detect([n3, n4, n5]),
            "cls": self.cls_head(n5),
            "seg": self.seg_head(p2, [n3, n4, n5], output_size=x.shape[-2:])
        }

# ==========================================
# 6. Unified Heads (Shared across models)
# ==========================================

class DetectionHead(nn.Module):
    def __init__(self, nc: int, ch: tuple[int, int, int], reg_max: int = 1):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        c_box = max(16, ch[0] // 4, self.reg_max * 4)
        c_cls = max(ch[0], min(self.nc, 100))

        self.box_branches = nn.ModuleList(
            nn.Sequential(Conv(x, c_box, 3), Conv(c_box, c_box, 3), nn.Conv2d(c_box, 4 * self.reg_max, 1)) 
            for x in ch
        )
        self.cls_branches = nn.ModuleList(
            nn.Sequential(Conv(x, c_cls, 3), Conv(c_cls, c_cls, 3), nn.Conv2d(c_cls, self.nc, 1)) 
            for x in ch
        )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for branch in self.cls_branches:
            nn.init.constant_(branch[-1].bias, bias_value)

    def forward(self, neck_features: list[torch.Tensor]):
        bs = neck_features[0].shape[0]
        boxes = torch.cat([self.box_branches[i](feat).view(bs, 4 * self.reg_max, -1) for i, feat in enumerate(neck_features)], dim=-1)
        scores = torch.cat([self.cls_branches[i](feat).view(bs, self.nc, -1) for i, feat in enumerate(neck_features)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": neck_features}

class ClsHead(nn.Module):
    def __init__(self, nc: int, in_ch: int):
        super().__init__()
        self.nc = nc
        self.cls_head = nn.Sequential(
            Conv(in_ch, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, nc)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(x)

class SegHead(nn.Module):
    def __init__(self, p2_ch: int, neck_ch: tuple[int, int, int], seg_ch: int = 64):
        super().__init__()
        self.p2_reduce = Conv(p2_ch, seg_ch, k=1, s=1)
        self.neck_reduce = nn.ModuleList(Conv(ch, seg_ch, k=1, s=1) for ch in neck_ch)
        self.seg_conv = nn.Sequential(
            Conv(seg_ch * 4, seg_ch * 2, 3),
            Conv(seg_ch * 2, seg_ch, k=3, s=1),
            nn.Conv2d(seg_ch, 1, kernel_size=1)
        )

    def forward(self, p2: torch.Tensor, neck_features: list[torch.Tensor], output_size: tuple[int, int] | None = None) -> torch.Tensor:
        target_size = p2.shape[-2:]
        p2_reduced = self.p2_reduce(p2)
        resized_features = [
            F.interpolate(reduce(feat), size=target_size, mode="nearest")
            for reduce, feat in zip(self.neck_reduce, neck_features)
        ]
        fused = torch.cat([p2_reduced, *resized_features], dim=1)
        mask_stride4 = self.seg_conv(fused)
        if output_size is None:
            output_size = (target_size[0] * 4, target_size[1] * 4)
        return F.interpolate(mask_stride4, size=output_size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    print("Testing Three Distinct Architectures with dummy input...")
    x = torch.randn(2, 3, 480, 640)
    
    models = {
        "RGB_Base (Control)": RGB_Base(num_classes=10),
        "RGB_Dynamic (SK Convs)": RGB_Dynamic(num_classes=10),
        "RGB_Attention (CoordAtt + GatedNeck)": RGB_Attention(num_classes=10)
    }
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        out = model(x)
        print(f"  Det Boxes: {out['det']['boxes'].shape}")
        print(f"  Det Scores: {out['det']['scores'].shape}")
        print(f"  Cls Out: {out['cls'].shape}")
        print(f"  Seg Out: {out['seg'].shape}")
        
        # Simple parameter count check
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params / 1e6:.2f}M")

        torchinfo.summary(model, input_size=(1, 3, 480, 640))