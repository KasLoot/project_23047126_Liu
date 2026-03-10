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



class Backbone_V1(nn.Module):
    """Simple CNN backbone for RGB_V1."""
    def __init__(self):
        super().__init__()
        self.channels = {"b1": 128, "b2": 128, "b3": 256}
        self.conv1 = Conv(3, 32, 3, 2)   # /2
        self.conv2 = Conv(32, 64, 3, 2)  # /4
        self.conv3 = Conv(64, self.channels["b1"], 3, 2)   # /8
        self.conv4 = Conv(self.channels["b1"], self.channels["b2"], 3, 2)   # /16
        self.conv5 = Conv(self.channels["b2"], self.channels["b3"], 3, 2)   # /32

    def forward(self, x: torch.Tensor, return_b0: bool = False):
        x = self.conv1(x)
        b0 = self.conv2(x)
        b1 = self.conv3(x)
        b2 = self.conv4(b1)
        b3 = self.conv5(b2)

        if return_b0:
            return b0, b1, b2, b3
        else:
            return b1, b2, b3






