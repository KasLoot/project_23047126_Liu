"""
Standalone YOLO26 Multi-Task Model for Hand Gesture Recognition.

A from-scratch implementation of the YOLO26 detect architecture with three task
heads: detection, classification, and segmentation. Designed for readability and
easy modification — not weight-compatible with official Ultralytics checkpoints.

Architecture overview:
    Input Image
         │
    ┌────▼────┐
    │ Backbone │  Conv stems → C3k2 blocks → SPPF → C2PSA
    └────┬────┘
         │  (P3, P4, P5 feature pyramids)
    ┌────▼────┐
    │   Neck   │  PAN/FPN with cross-attention refinement
    └──┬─┬─┬──┘
       │ │ │   (N3, N4, N5 multi-scale features)
    ┌──▼─▼─▼──────────────────────────────┐
    │        Three Task Heads              │
    │  ┌─────────┬──────────┬───────────┐  │
    │  │  Detect  │ Classify │ Segment   │  │
    │  │ (N3,4,5) │   (N5)   │(N3+N4+N5)│  │
    │  └─────────┴──────────┴───────────┘  │
    └──────────────────────────────────────┘

Compound scaling: n / s / m / l / x variants control depth, width, and max channels.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  Scale Configuration
# =====================================================================

@dataclass
class ScaleConfig:
    """Compound-scaling multipliers for a single model size variant."""
    depth_multiplier: float
    width_multiplier: float
    max_channels: int


SCALE_CONFIGS: dict[str, ScaleConfig] = {
    "n": ScaleConfig(depth_multiplier=0.50, width_multiplier=0.25, max_channels=1024),
    "s": ScaleConfig(depth_multiplier=0.50, width_multiplier=0.50, max_channels=1024),
    "m": ScaleConfig(depth_multiplier=0.50, width_multiplier=1.00, max_channels=512),
    "l": ScaleConfig(depth_multiplier=1.00, width_multiplier=1.00, max_channels=512),
    "x": ScaleConfig(depth_multiplier=1.00, width_multiplier=1.50, max_channels=512),
}


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round ``value`` to the nearest multiple of ``divisor``."""
    return int((value + divisor / 2) // divisor * divisor)


def scale_channels(base_channels: int, width: float, max_channels: int) -> int:
    """Apply width scaling to a base channel count, clamped by ``max_channels``."""
    return make_divisible(min(base_channels, max_channels) * width, 8)


def scale_depth(base_repeats: int, depth: float) -> int:
    """Apply depth scaling to a base repeat count (minimum 1 if base > 1)."""
    return max(round(base_repeats * depth), 1) if base_repeats > 1 else base_repeats


# =====================================================================
#  Core Building Blocks
# =====================================================================

def _auto_pad(kernel_size: int, padding: int | None = None) -> int:
    """Return same-padding value for a given kernel size, or pass through explicit padding."""
    return kernel_size // 2 if padding is None else padding


class ConvBnAct(nn.Module):
    """Convolution → BatchNorm → Activation (SiLU by default).

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Convolution kernel size.
        stride:       Convolution stride.
        padding:      Explicit padding (``None`` → same-padding).
        groups:       Grouped convolution factor.
        activation:   ``True`` for SiLU, an ``nn.Module`` instance, or ``False`` / ``None`` for identity.
    """

    DEFAULT_ACTIVATION = nn.SiLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: bool | nn.Module = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            _auto_pad(kernel_size, padding), groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation is True:
            self.act = self.DEFAULT_ACTIVATION()
        elif isinstance(activation, nn.Module):
            self.act = activation
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck: 3×3 → 3×3 with an optional skip connection.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        use_shortcut: Add residual connection when ``in_channels == out_channels``.
        expansion:    Hidden-channel expansion ratio relative to ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBnAct(in_channels, hidden_channels, kernel_size=3)
        self.conv2 = ConvBnAct(hidden_channels, out_channels, kernel_size=3)
        self.has_shortcut = use_shortcut and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.has_shortcut else out


# =====================================================================
#  CSP-style Blocks (C2f / C3k / C3k2)
# =====================================================================

class C2f(nn.Module):
    """C2f Cross-Stage-Partial block.

    Splits an input into two halves along channels, runs one half through a
    chain of ``Bottleneck`` blocks, then concatenates all intermediate outputs
    and projects back to ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        use_shortcut: bool = False,
        expansion: float = 0.5,
    ):
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.input_proj = ConvBnAct(in_channels, 2 * self.hidden_channels, kernel_size=1)
        self.output_proj = ConvBnAct((2 + num_blocks) * self.hidden_channels, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            Bottleneck(self.hidden_channels, self.hidden_channels, use_shortcut=use_shortcut, expansion=1.0)
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = list(self.input_proj(x).chunk(2, dim=1))
        for block in self.blocks:
            chunks.append(block(chunks[-1]))
        return self.output_proj(torch.cat(chunks, dim=1))


class C3k(nn.Module):
    """Stack of ``Bottleneck`` blocks with full-width expansion (kernel-like behavior)."""

    def __init__(self, channels: int, num_blocks: int = 2, use_shortcut: bool = True):
        super().__init__()
        self.blocks = nn.Sequential(
            *(Bottleneck(channels, channels, use_shortcut=use_shortcut, expansion=1.0) for _ in range(num_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class C3k2(C2f):
    """C3k2 block — a C2f variant that optionally swaps ``Bottleneck`` for ``C3k``.

    When ``use_c3k=True``, each repeated block is a deeper C3k stack rather
    than a single Bottleneck. This increases capacity without changing the
    outer CSP structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        use_c3k: bool = False,
        expansion: float = 0.5,
        use_shortcut: bool = True,
    ):
        super().__init__(in_channels, out_channels, num_blocks=num_blocks, use_shortcut=use_shortcut, expansion=expansion)
        # Override the block list from C2f with either C3k or Bottleneck
        self.blocks = nn.ModuleList(
            C3k(self.hidden_channels, num_blocks=2, use_shortcut=use_shortcut) if use_c3k
            else Bottleneck(self.hidden_channels, self.hidden_channels, use_shortcut=use_shortcut, expansion=1.0)
            for _ in range(num_blocks)
        )


# =====================================================================
#  Spatial Pyramid & Attention
# =====================================================================

class SPPF(nn.Module):
    """Fast Spatial Pyramid Pooling: cascaded max-pools at a single kernel size.

    Applies ``num_pools`` successive max-pool operations, concatenates all
    intermediate results with the input, and projects to ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_kernel: int = 5, num_pools: int = 3):
        super().__init__()
        mid_channels = in_channels // 2
        self.reduce = ConvBnAct(in_channels, mid_channels, kernel_size=1, activation=False)
        self.project = ConvBnAct(mid_channels * (num_pools + 1), out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
        self.num_pools = num_pools

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [self.reduce(x)]
        for _ in range(self.num_pools):
            features.append(self.pool(features[-1]))
        return self.project(torch.cat(features, dim=1))


class SelfAttention(nn.Module):
    """Multi-head self-attention with depthwise positional encoding.

    Projects input to queries, keys, and values via a single fused 1×1 conv,
    then applies scaled dot-product attention and adds a learned depthwise
    positional bias.
    """

    def __init__(self, channels: int, num_heads: int = 4, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.head_dim = channels // self.num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim ** -0.5

        qkv_channels = channels + self.key_dim * self.num_heads * 2
        self.qkv_proj = ConvBnAct(channels, qkv_channels, kernel_size=1, activation=False)
        self.output_proj = ConvBnAct(channels, channels, kernel_size=1, activation=False)
        self.positional_encoding = ConvBnAct(channels, channels, kernel_size=3, groups=channels, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        num_tokens = height * width

        qkv = self.qkv_proj(x)
        q, k, v = qkv.view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, num_tokens).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn_weights = (q.transpose(-2, -1) @ k) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        attended = (v @ attn_weights.transpose(-2, -1)).view(batch, channels, height, width)
        positional_bias = self.positional_encoding(v.reshape(batch, channels, height, width))
        return self.output_proj(attended + positional_bias)


class CrossAttention(nn.Module):
    """Multi-head cross-attention between different spatial feature maps.

    Queries, keys, and values can have different channel dimensions. All are
    projected to a common ``out_dim`` (= query channels) before attention.
    """

    def __init__(self, qkv_dims: list[int], num_heads: int = 4, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        out_dim = qkv_dims[0]

        if out_dim % self.num_heads != 0:
            raise ValueError(f"Output dim {out_dim} must be divisible by num_heads {self.num_heads}")

        self.query_proj = ConvBnAct(qkv_dims[0], out_dim, kernel_size=1, activation=False)
        self.key_proj = ConvBnAct(qkv_dims[1], out_dim, kernel_size=1, activation=False)
        self.value_proj = ConvBnAct(qkv_dims[2], out_dim, kernel_size=1, activation=False)
        self.output_proj = ConvBnAct(out_dim, out_dim, kernel_size=1, activation=False)
        self.positional_encoding = ConvBnAct(out_dim, out_dim, kernel_size=3, groups=out_dim, activation=False)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, C, H, W) → (B, heads, N, head_dim)."""
        batch, channels, h, w = x.shape
        head_dim = channels // self.num_heads
        return x.reshape(batch, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

    def _from_heads(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reshape (B, heads, N, head_dim) → (B, C, H, W)."""
        batch, heads, _, head_dim = x.shape
        return x.permute(0, 1, 3, 2).contiguous().view(batch, heads * head_dim, height, width)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        _, _, h_q, w_q = query.shape

        q = self._to_heads(self.query_proj(query))
        k = self._to_heads(self.key_proj(key))
        v = self._to_heads(self.value_proj(value))

        scale = q.shape[-1] ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)

        attended = self._from_heads(attn_weights @ v, h_q, w_q)
        positional_bias = self.positional_encoding(self._from_heads(q, h_q, w_q))
        return self.output_proj(attended + positional_bias)


# =====================================================================
#  PSA Blocks (Position-Sensitive Attention)
# =====================================================================

class PSABlock(nn.Module):
    """Self-attention + feed-forward with residual connections."""

    def __init__(self, channels: int):
        super().__init__()
        num_heads = max(channels // 64, 1)
        self.attention = SelfAttention(channels, num_heads=num_heads, attn_ratio=0.5)
        self.feed_forward = nn.Sequential(
            ConvBnAct(channels, channels * 2, kernel_size=1),
            ConvBnAct(channels * 2, channels, kernel_size=1, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x


class C2PSA(nn.Module):
    """CSP wrapper around stacked PSA blocks.

    Splits the input into two channel halves. One half passes through a chain
    of PSA blocks, then both halves are concatenated and projected.
    """

    def __init__(self, channels: int, out_channels: int, num_blocks: int = 1, expansion: float = 0.5):
        super().__init__()
        assert channels == out_channels, "C2PSA requires in_channels == out_channels"
        self.hidden_channels = int(channels * expansion)
        self.input_proj = ConvBnAct(channels, 2 * self.hidden_channels, kernel_size=1)
        self.output_proj = ConvBnAct(2 * self.hidden_channels, channels, kernel_size=1)
        self.psa_blocks = nn.Sequential(*(PSABlock(self.hidden_channels) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        passthrough, attended = self.input_proj(x).split(self.hidden_channels, dim=1)
        attended = self.psa_blocks(attended)
        return self.output_proj(torch.cat((passthrough, attended), dim=1))


# =====================================================================
#  Backbone + Neck
# =====================================================================

class BackboneWithNeck(nn.Module):
    """YOLO26 backbone (feature extractor) and PAN/FPN neck (feature aggregator).

    Produces three multi-scale feature maps [N3, N4, N5] at strides 8, 16, 32.

    Backbone stages:
        stem         → stride 4  (2× downsamples via two Conv layers)
        stage_p3     → stride 8  (produces P3 features)
        stage_p4     → stride 16 (produces P4 features)
        stage_p5     → stride 32 (produces P5 features, refined by SPPF + C2PSA)

    Neck merges features top-down and bottom-up with cross-attention refinement.
    """

    def __init__(
        self,
        num_classes: int = 10,
        scale: Literal["n", "s", "m", "l", "x"] = "n",
        end2end: bool = True,
        reg_max: int = 1,
    ):
        super().__init__()
        if scale not in SCALE_CONFIGS:
            raise ValueError(f"Unknown scale '{scale}'. Choose from {list(SCALE_CONFIGS)}")

        cfg = SCALE_CONFIGS[scale]
        depth = cfg.depth_multiplier
        width = cfg.width_multiplier
        max_ch = cfg.max_channels

        # --- Compute scaled channel widths ---
        self.ch_64 = scale_channels(64, width, max_ch)    # Stem output
        self.ch_128 = scale_channels(128, width, max_ch)   # After second stem conv
        self.ch_256 = scale_channels(256, width, max_ch)   # P3 level
        self.ch_512 = scale_channels(512, width, max_ch)   # P4 level
        self.ch_1024 = scale_channels(1024, width, max_ch) # P5 level

        num_repeats = scale_depth(2, depth)

        # ── Backbone ──

        # Stem: two strided convolutions (image → stride-4 features)
        self.stem_conv1 = ConvBnAct(3, self.ch_64, kernel_size=3, stride=2)
        self.stem_conv2 = ConvBnAct(self.ch_64, self.ch_128, kernel_size=3, stride=2)
        self.stem_csp = C3k2(self.ch_128, self.ch_256, num_blocks=num_repeats, use_c3k=False, expansion=0.25, use_shortcut=False)

        # Stage P3: stride 8
        self.p3_downsample = ConvBnAct(self.ch_256, self.ch_256, kernel_size=3, stride=2)
        self.p3_csp = C3k2(self.ch_256, self.ch_512, num_blocks=num_repeats, use_c3k=False, expansion=0.25, use_shortcut=False)

        # Stage P4: stride 16
        self.p4_downsample = ConvBnAct(self.ch_512, self.ch_512, kernel_size=3, stride=2)
        self.p4_csp = C3k2(self.ch_512, self.ch_512, num_blocks=num_repeats, use_c3k=True, expansion=0.5, use_shortcut=True)

        # Stage P5: stride 32 (deepest features)
        self.p5_downsample = ConvBnAct(self.ch_512, self.ch_1024, kernel_size=3, stride=2)
        self.p5_csp = C3k2(self.ch_1024, self.ch_1024, num_blocks=num_repeats, use_c3k=True, expansion=0.5, use_shortcut=True)
        self.p5_sppf = SPPF(self.ch_1024, self.ch_1024, pool_kernel=5, num_pools=3)
        self.p5_psa = C2PSA(self.ch_1024, self.ch_1024, num_blocks=num_repeats)

        # ── Neck (PAN/FPN) ──

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down path: P5 → P4 → P3
        self.neck_td_p4_fuse = C3k2(self.ch_1024 + self.ch_512, self.ch_512, num_blocks=num_repeats, use_c3k=True, expansion=0.5, use_shortcut=True)
        self.neck_td_p3_fuse = C3k2(self.ch_512 + self.ch_512, self.ch_256, num_blocks=num_repeats, use_c3k=True, expansion=0.5, use_shortcut=True)

        # Bottom-up path: N3 → N4 → N5
        self.neck_bu_n3_to_n4_down = ConvBnAct(self.ch_256, self.ch_256, kernel_size=3, stride=2)
        self.neck_bu_n4_fuse = C3k2(self.ch_256 + self.ch_512, self.ch_512, num_blocks=num_repeats, use_c3k=True, expansion=0.5, use_shortcut=True)
        self.neck_bu_n4_to_n5_down = ConvBnAct(self.ch_512, self.ch_512, kernel_size=3, stride=2)
        self.neck_bu_n5_fuse = C3k2(self.ch_512 + self.ch_1024, self.ch_1024, num_blocks=1, use_c3k=True, expansion=0.5, use_shortcut=True)

        # Cross-attention refinement in the neck
        self.neck_n4_cross_attn = CrossAttention(
            qkv_dims=[self.ch_512, self.ch_512, self.ch_512],
            num_heads=max(self.ch_512 // 32, 1),
        )
        self.neck_n5_cross_attn = CrossAttention(
            qkv_dims=[self.ch_1024, self.ch_512, self.ch_512],
            num_heads=max(self.ch_1024 // 32, 1),
        )

        # Store config for reference
        self.num_classes = num_classes
        self.scale = scale
        self.end2end = end2end
        self.reg_max = reg_max

    # Expose channel sizes for downstream heads (using the names the rest of the code expects)
    @property
    def c3(self) -> int:
        return self.ch_256

    @property
    def c4(self) -> int:
        return self.ch_512

    @property
    def c5(self) -> int:
        return self.ch_1024

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run backbone + neck, returning [N3, N4, N5] feature maps."""

        # ── Backbone ──
        x = self.stem_conv1(x)
        x = self.stem_conv2(x)
        x = self.stem_csp(x)

        p3 = self.p3_csp(self.p3_downsample(x))
        p4 = self.p4_csp(self.p4_downsample(p3))

        p5 = self.p5_downsample(p4)
        p5 = self.p5_csp(p5)
        p5 = self.p5_sppf(p5)
        p5 = self.p5_psa(p5)

        # ── Neck: top-down ──
        n4 = self.neck_td_p4_fuse(torch.cat([self.upsample(p5), p4], dim=1))
        n4 = self.neck_n4_cross_attn(query=n4, key=p3, value=p3)
        n3 = self.neck_td_p3_fuse(torch.cat([self.upsample(n4), p3], dim=1))

        # ── Neck: bottom-up ──
        n4_out = self.neck_bu_n4_fuse(torch.cat([self.neck_bu_n3_to_n4_down(n3), n4], dim=1))
        n5_out = self.neck_bu_n5_fuse(torch.cat([self.neck_bu_n4_to_n5_down(n4_out), p5], dim=1))
        n5_out = self.neck_n5_cross_attn(query=n5_out, key=p4, value=p4)

        return [n3, n4_out, n5_out]


# =====================================================================
#  Detection Head
# =====================================================================

class DetectionHead(nn.Module):
    """YOLO26-style detection head with optional dual-branch (one-to-many + one-to-one).

    Each branch has separate box-regression and classification sub-heads applied
    to every scale level independently, then concatenated across spatial positions.

    Args:
        num_classes:       Number of object classes.
        feature_channels:  Channel counts for each neck output level, e.g. (256, 512, 1024).
        reg_max:           Regression granularity (1 = direct 4-dim box distances).
        end2end:           If True, create a second one-to-one branch for NMS-free inference.
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_channels: tuple[int, ...] = (256, 512, 1024),
        reg_max: int = 1,
        end2end: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(feature_channels)
        self.reg_max = reg_max
        self.outputs_per_anchor = num_classes + 4 * reg_max
        self.end2end = end2end
        self.max_detections = 300

        # Determine hidden-channel widths for each sub-head
        box_hidden = max(16, feature_channels[0] // 4, reg_max * 4)
        cls_hidden = max(feature_channels[0], min(num_classes, 100))

        # One-to-many branch (dense supervision during training)
        self.box_heads = nn.ModuleList(
            nn.Sequential(
                ConvBnAct(ch, box_hidden, kernel_size=3),
                ConvBnAct(box_hidden, box_hidden, kernel_size=3),
                nn.Conv2d(box_hidden, 4 * reg_max, kernel_size=1),
            )
            for ch in feature_channels
        )
        self.cls_heads = nn.ModuleList(
            nn.Sequential(
                ConvBnAct(ch, cls_hidden, kernel_size=3),
                ConvBnAct(cls_hidden, cls_hidden, kernel_size=3),
                nn.Conv2d(cls_hidden, num_classes, kernel_size=1),
            )
            for ch in feature_channels
        )

        # One-to-one branch (NMS-free end-to-end inference)
        if end2end:
            self.one2one_box_heads = copy.deepcopy(self.box_heads)
            self.one2one_cls_heads = copy.deepcopy(self.cls_heads)

    def _run_branch(
        self,
        features: list[torch.Tensor],
        box_heads: nn.ModuleList,
        cls_heads: nn.ModuleList,
    ) -> dict[str, torch.Tensor]:
        """Apply box + cls sub-heads to each scale level and concatenate."""
        batch_size = features[0].shape[0]
        boxes = torch.cat(
            [box_heads[i](features[i]).view(batch_size, 4 * self.reg_max, -1) for i in range(self.num_levels)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_heads[i](features[i]).view(batch_size, self.num_classes, -1) for i in range(self.num_levels)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": features}

    def _postprocess_top_k(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Select top-k detections per image (NMS-free). Returns (B, K, 6): [x, y, w, h, conf, cls]."""
        batch, _, num_anchors = boxes.shape
        boxes = boxes.permute(0, 2, 1)                      # (B, A, 4)
        scores = scores.sigmoid().permute(0, 2, 1)          # (B, A, nc)

        confidence, class_idx = scores.max(dim=-1, keepdim=True)  # (B, A, 1)
        k = min(self.max_detections, num_anchors)
        topk_conf, topk_indices = confidence.squeeze(-1).topk(k, dim=1)

        # Gather boxes and class indices for top-k
        expand_4 = topk_indices.unsqueeze(-1).expand(batch, k, 4)
        expand_1 = topk_indices.unsqueeze(-1)

        topk_boxes = boxes.gather(1, expand_4)
        topk_classes = class_idx.gather(1, expand_1).float()
        topk_conf = topk_conf.unsqueeze(-1)

        return torch.cat([topk_boxes, topk_conf, topk_classes], dim=-1)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._run_branch(features, self.box_heads, self.cls_heads)


# =====================================================================
#  Multi-Task Model (Detection + Classification + Segmentation)
# =====================================================================

class HandGestureMultiTask(nn.Module):
    """Multi-task hand gesture model with three output heads.

    Heads:
        det:  Object detection (bounding boxes + class scores per anchor).
        cls:  Image-level gesture classification (from deepest features).
        seg:  Binary hand segmentation mask (from fused multi-scale features).

    Args:
        num_classes:       Number of gesture classes.
        end2end:           Enable dual-branch detection for NMS-free inference.
        scale:             Compound-scaling variant.
        reg_max:           Detection regression granularity.
    """

    def __init__(
        self,
        num_classes: int = 10,
        end2end: bool = True,
        scale: Literal["n", "s", "m", "l", "x"] = "n",
        reg_max: int = 1,
    ):
        super().__init__()

        # Shared backbone + neck
        self.backbone = BackboneWithNeck(num_classes=num_classes, scale=scale, end2end=end2end, reg_max=reg_max)

        feature_channels = (self.backbone.c3, self.backbone.c4, self.backbone.c5)

        # ── Detection Head ──
        self.detect_head = DetectionHead(
            num_classes=num_classes,
            feature_channels=feature_channels,
            reg_max=reg_max,
            end2end=end2end,
        )

        c3_ch, c4_ch, c5_ch = feature_channels

        # ── Classification Head ──
        # Operates on N5 (deepest/most semantic features)
        self.classify_head = nn.Sequential(
            ConvBnAct(c5_ch, 256, kernel_size=3),
            ConvBnAct(256, 256, kernel_size=3, stride=2),  # Spatial downsampling
            ConvBnAct(256, 256, kernel_size=3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # ── Segmentation Head ──
        # Operates on fused N3 + upsampled(N4) + upsampled(N5)
        fused_channels = c3_ch + c4_ch + c5_ch
        self.segment_head = nn.ModuleList([
            ConvBnAct(fused_channels, 256, kernel_size=3),
            ConvBnAct(256, 256, kernel_size=3),
            ConvBnAct(256, 256, kernel_size=3),
            ConvBnAct(256, 256, kernel_size=3),
            ConvBnAct(256, 256, kernel_size=3),
            nn.Conv2d(256, 1, kernel_size=1),  # Binary mask output
        ])

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for fine-tuning heads only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        tasks: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """Run selected task heads.

        Args:
            x:     Input images, shape (B, 3, H, W).
            tasks: Subset of ("det", "cls", "seg") to compute. Defaults to all three.

        Returns:
            Dictionary mapping task name → output tensor(s).
        """
        if tasks is None:
            tasks = ("det", "cls", "seg")

        n3, n4, n5 = self.backbone(x)
        outputs: dict[str, torch.Tensor | dict] = {}

        if "det" in tasks:
            outputs["det"] = self.detect_head([n3, n4, n5])

        if "cls" in tasks:
            outputs["cls"] = self.classify_head(n5)

        if "seg" in tasks:
            # Upsample N4 and N5 to match N3's spatial resolution, then fuse
            target_size = n3.shape[-2:]
            n4_upsampled = F.interpolate(n4, size=target_size, mode="nearest")
            n5_upsampled = F.interpolate(n5, size=target_size, mode="nearest")
            fused = torch.cat([n3, n4_upsampled, n5_upsampled], dim=1)

            for layer in self.segment_head:
                fused = layer(fused)

            # Upsample low-res mask to original input resolution
            outputs["seg"] = F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return outputs


# =====================================================================
#  Demo / Smoke Test
# =====================================================================

def _demo() -> None:
    """Quick smoke test: build the model, run a dummy forward pass, and print output shapes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HandGestureMultiTask().to(device)
    model.eval()

    try:
        import torchinfo
        torchinfo.summary(model, input_size=(1, 3, 256, 256), device=str(device), depth=7)
    except ImportError:
        print("(Install torchinfo for a detailed model summary)")

    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        preds = model(x)

    det = preds["det"]
    print(f"Detection boxes:  {det['boxes'].shape}")
    print(f"Detection scores: {det['scores'].shape}")
    print(f"Detection feats:  {[f.shape for f in det['feats']]}")
    print(f"Classification:   {preds['cls'].shape}")
    print(f"Segmentation:     {preds['seg'].shape}")

    print("\nCross-attention smoke test:")
    attn = CrossAttention(qkv_dims=[128, 64, 64], num_heads=4).to(device)
    q = torch.randn(1, 128, 16, 16, device=device)
    k = torch.randn(1, 64, 32, 32, device=device)
    v = torch.randn(1, 64, 32, 32, device=device)
    print(f"Output shape: {attn(q, k, v).shape}")


if __name__ == "__main__":
    _demo()