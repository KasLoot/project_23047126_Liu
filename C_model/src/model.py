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






class BackBone(nn.Module):
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

        # Expose channel dims for downstream heads and the neck.
        self.channels = {"p2": c3, "p3": c4, "p4": c4, "p5": c5}

    def forward(
        self,
        x: torch.Tensor,
        return_p2: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        if return_p2:
            return p2, p3, p4, p5
        return p3, p4, p5



# class Neck(nn.Module):
#     """
#     Feature Pyramid Network (FPN) & Path Aggregation Network (PAN).
#     Fuses features from different backbone stages.
#     """
#     def __init__(self, in_channels: dict[str, int]):
#         super().__init__()
#         c3, c4, c5 = 64, 128, 256  # Hardcoded Nano channels (p3 actually outputs 128 channels in backbone)
#         # Note: Backbone p3 outputs c4(128). We respect this flow.
#         p3_ch, p4_ch, p5_ch = in_channels["p3"], in_channels["p4"], in_channels["p5"]

#         self.up = nn.Upsample(scale_factor=2, mode="nearest")

#         # --- Top-Down Pathway (FPN) ---
#         self.fpn_p4 = C3k2(p5_ch + p4_ch, c4, n=1, c3k=True, e=0.5, shortcut=True)
#         self.n4_attn = CrossAttention(qkv_dim=[c4, p3_ch, p3_ch], num_heads=max(c4 // 32, 1))
        
#         self.fpn_p3 = C3k2(c4 + p3_ch, c3, n=1, c3k=True, e=0.5, shortcut=True)

#         # --- Bottom-Up Pathway (PAN) ---
#         self.down_n3 = Conv(c3, c3, k=3, s=2)
#         self.pan_n4 = C3k2(c3 + c4, c4, n=1, c3k=True, e=0.5, shortcut=True)

#         self.down_n4 = Conv(c4, c4, k=3, s=2)
#         self.pan_n5 = C3k2(c4 + p5_ch, c5, n=1, c3k=True, e=0.5, shortcut=True)
#         self.n5_attn = CrossAttention(qkv_dim=[c5, p4_ch, p4_ch], num_heads=max(c5 // 32, 1))

#         # Expose final channel dims for the Heads to use
#         self.channels = {"n3": c3, "n4": c4, "n5": c5}

#     def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
#         # Top-Down (FPN)
#         n4 = self.fpn_p4(torch.cat([self.up(p5), p4], dim=1))
#         n4 = n4 + self.n4_attn(n4, p3, p3)
#         n3 = self.fpn_p3(torch.cat([self.up(n4), p3], dim=1))

#         # Bottom-Up (PAN)
#         n4_out = self.pan_n4(torch.cat([self.down_n3(n3), n4], dim=1))
#         n5_out = self.pan_n5(torch.cat([self.down_n4(n4_out), p5], dim=1))
#         n5_out = n5_out + self.n5_attn(n5_out, p4, p4)

#         return n3, n4_out, n5_out


class Neck(nn.Module):
    def __init__(self, in_channels: dict[str, int]):
        super().__init__()
        # Initialize the neck with the input channels from the backbone
        self.in_channels = in_channels
        p3_ch, p4_ch, p5_ch = in_channels["p3"], in_channels["p4"], in_channels["p5"]
        self.c3, self.c4, self.c5 = 64, 128, 256

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=4, mode="nearest")
        self.down_1 = Conv(p3_ch, p3_ch, k=3, s=2)
        self.down_2 = Conv(p3_ch, p3_ch, k=3, s=4)
        self.down_3 = Conv(p4_ch, p4_ch, k=3, s=2)

        self.n1_attn_34 = CrossAttention(qkv_dim=[p3_ch, p4_ch, p4_ch])
        self.n1_attn_35 = CrossAttention(qkv_dim=[p3_ch, p5_ch, p5_ch])
        self.n1_bn = nn.BatchNorm2d(p3_ch)
        self.n1_fpn = C3k2(p3_ch + p4_ch + p5_ch, self.c3, n=1, c3k=True, e=0.5, shortcut=True)
        

        self.n2_attn_43 = CrossAttention(qkv_dim=[p4_ch, p3_ch, p3_ch])
        self.n2_attn_45 = CrossAttention(qkv_dim=[p4_ch, p5_ch, p5_ch])
        self.n2_bn = nn.BatchNorm2d(p4_ch)
        self.n2_fpn = C3k2(p4_ch + p3_ch + p5_ch, self.c4, n=1, c3k=True, e=0.5, shortcut=True)
        

        self.n3_attn_53 = CrossAttention(qkv_dim=[p5_ch, p3_ch, p3_ch])
        self.n3_attn_54 = CrossAttention(qkv_dim=[p5_ch, p4_ch, p4_ch])
        self.n3_bn = nn.BatchNorm2d(p5_ch)
        self.n3_fpn = C3k2(p5_ch + p3_ch + p4_ch, self.c5, n=1, c3k=True, e=0.5, shortcut=True)



    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Implementation for the neck forward pass
        n1 = self.n1_bn(p3 + self.n1_attn_35(self.n1_attn_34(p3, p4, p4), p5, p5))
        n1 = self.n1_fpn(torch.cat([n1, self.up2(p4), self.up4(p5)], dim=1))
        # print(f"n1 shape: {n1.shape}")  # Debug print

        n2 = self.n2_bn(p4 + self.n2_attn_45(self.n2_attn_43(p4, p3, p3), p5, p5))
        # print(f"n2 shape: {n2.shape}, down_1(p3) shape: {self.down_1(p3).shape}, up2(p5) shape: {self.up2(p5).shape}")  # Debug print
        n2 = self.n2_fpn(torch.cat([n2, self.down_1(p3), self.up2(p5)], dim=1))
        # print(f"n2 shape after FPN: {n2.shape}")  # Debug print

        n3 = self.n3_bn(p5 + self.n3_attn_54(self.n3_attn_53(p5, p3, p3), p4, p4))
        # print(f"n3 shape before FPN: {n3.shape}, self.down_2(p3) shape: {self.down_2(p3).shape}, self.down_3(p4) shape: {self.down_3(p4).shape}")  # Debug print
        n3 = self.n3_fpn(torch.cat([n3, self.down_2(p3), self.down_3(p4)], dim=1))
        # print(f"n3 shape after FPN: {n3.shape}")  # Debug print


        return n1, n2, n3

        

class DetectionHead(nn.Module):
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



class ClsHead(nn.Module):
    """Global Image Classification Head."""
    def __init__(self, nc: int, in_ch: int):
        super().__init__()
        self.nc = nc
        self.cls_head = nn.Sequential(
            # 1. Process the spatial layout while it still exists
            Conv(in_ch, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            
            # 2. Now that we've learned complex spatial patterns, squash it
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 3. Deeper fully connected network with stronger dropout
            nn.Dropout(0.3), # Increased dropout to prevent overfitting this large head
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, nc)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cls_head(x)
        return x



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

    def forward(
        self,
        p2: torch.Tensor,
        neck_features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
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



class HandGestureMultiTask(nn.Module):
    """Unified Multi-Task Model for Detection, Classification, and Segmentation."""
    def __init__(self, num_classes: int, reg_max: int = 1):
        super().__init__()
        self.backbone = BackBone()
        self.neck = Neck(in_channels=self.backbone.channels)
        neck_ch = (self.neck.c3, self.neck.c4, self.neck.c5)
        self.detect = DetectionHead(nc=num_classes, ch=neck_ch, reg_max=reg_max)
        self.cls_head = ClsHead(nc=num_classes, in_ch=neck_ch[2])  # Use the deepest neck feature for classification
        self.seg_head = SegHead(p2_ch=self.backbone.channels["p2"], neck_ch=neck_ch)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p2, p3, p4, p5 = self.backbone(x, return_p2=True)
        n3, n4, n5 = self.neck(p3, p4, p5)
        
        det_outputs = self.detect([n3, n4, n5])
        cls_output = self.cls_head(n5)  # Use the deepest neck feature for classification
        seg_output = self.seg_head(p2, [n3, n4, n5], output_size=x.shape[-2:])

        return {
            "det": det_outputs,
            "cls": cls_output,
            "seg": seg_output
        }







if __name__ == "__main__":

    print("Testing BackBone with dummy input...")

    # Create a dummy input tensor (batch_size=1, channels=3, height=480, width=640)
    x = torch.randn(1, 3, 480, 640)
    backbone = BackBone()
    p2, p3, p4, p5 = backbone(x, return_p2=True)
    print(f"P2 shape: {p2.shape}")  # Expected: [1, 64, 120, 160]
    print(f"P3 shape: {p3.shape}")  # Expected: [1, 128, 60, 80]
    print(f"P4 shape: {p4.shape}")  # Expected: [1, 128, 30, 40]
    print(f"P5 shape: {p5.shape}")  # Expected: [1, 256, 15, 20]


    # Test the Neck with the output from the Backbone
    channels = backbone.channels
    neck = Neck(in_channels=channels)
    n1, n2, n3 = neck(p3, p4, p5)
    print(f"N1 shape: {n1.shape}")  # Expected: [1, 64, 60, 80]
    print(f"N2 shape: {n2.shape}")  # Expected: [1, 128, 30, 40]
    print(f"N3 shape: {n3.shape}")  # Expected: [1, 256, 15, 20]


    # Test the Detection Head with the output from the Neck
    det_head = DetectionHead(nc=10, ch=(64, 128, 256), reg_max=1)
    det_outputs = det_head([n1, n2, n3])
    print(f"Boxes shape: {det_outputs['boxes'].shape}")  # Expected: [1, 4, 60*80 + 30*40 + 15*20]
    print(f"Scores shape: {det_outputs['scores'].shape}")  # Expected: [1, 10, 60*80 + 30*40 + 15*20]


    # Test the Classification Head with the output from the Neck
    cls_head = ClsHead(nc=10, in_ch=256)
    cls_output = cls_head(n3)
    print(f"Classification output shape: {cls_output.shape}")  # Expected: [1, 10]

    # Test the Segmentation Head with the output from the Neck
    seg_head = SegHead(p2_ch=backbone.channels["p2"], neck_ch=(64, 128, 256))
    seg_output = seg_head(p2, [n1, n2, n3], output_size=x.shape[-2:])
    print(f"Segmentation output shape: {seg_output.shape}")  # Expected: [1, 1, 480, 640]


    # Test the full multi-task model
    print("Testing the full HandGestureMultiTask model with dummy input...")
    multi_task_model = HandGestureMultiTask(num_classes=10, reg_max=1)
    multi_task_outputs = multi_task_model(x)
    print(f"Multi-task outputs shapes:")
    for key, value in multi_task_outputs.items():
        print(f"  {key}: {value.shape}")
    