#!/usr/bin/env python3

"""Architecture-only implementation of MambaVision.

This module intentionally excludes:
- checkpoint loading helpers
- pretrained cfg handling
- model registry decorators / factory wrappers

It keeps only the model building blocks and the `MambaVision` network.
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from selective_scan import selective_scan_fn
from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_
from timm.models.vision_transformer import Mlp


# --------------------------------------------------------------------------------------
# Window helpers
# --------------------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature maps into local windows.

    Args:
        x: Input tensor with shape ``(B, C, H, W)``.
        window_size: Window edge length.

    Returns:
        Tensor with shape ``(num_windows * B, window_size * window_size, C)``.
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverse window partition back to feature map layout.

    Args:
        windows: Tensor with shape ``(num_windows * B, window_size * window_size, C)``.
        window_size: Window edge length.
        h: Output height.
        w: Output width.

    Returns:
        Tensor with shape ``(B, C, H, W)``.
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.reshape(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, windows.shape[2], h, w)
    return x


# --------------------------------------------------------------------------------------
# Core blocks
# --------------------------------------------------------------------------------------
class Downsample(nn.Module):
    """2x spatial downsampling via strided convolution."""

    def __init__(self, dim: int, keep_dim: bool = False) -> None:
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """Stem patch embedding with two strided conv layers."""

    def __init__(self, in_chans: int = 3, in_dim: int = 64, dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_down(self.proj(x))


class ConvBlock(nn.Module):
    """Residual Conv-BN-GELU-Conv-BN block with optional layer scaling."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)

        self.use_layer_scale = isinstance(layer_scale, (int, float))
        if self.use_layer_scale:
            self.gamma = nn.Parameter(float(layer_scale) * torch.ones(dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)

        return residual + self.drop_path(x)


class MambaVisionMixer(nn.Module):
    """Mamba mixer block used in non-attention transformer slots."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        a = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            # Keep parity with original implementation.
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            # Keep parity with original implementation.
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply Mamba mixer.

        Args:
            hidden_states: Tensor with shape ``(B, L, D)``.

        Returns:
            Tensor with shape ``(B, L, D)``.
        """
        _, seqlen, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        a = -torch.exp(self.A_log.float())

        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, b_mat, c_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        b_mat = rearrange(b_mat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        c_mat = rearrange(c_mat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            a,
            b_mat,
            c_mat,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class Attention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Transformer-style block with either attention or Mamba mixer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        counter: int,
        transformer_blocks: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_block: type[nn.Module] = Mlp,
        layer_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        use_layer_scale = isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(float(layer_scale) * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(float(layer_scale) * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """One stage of MambaVision (conv or windowed transformer/mamba blocks)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[bool] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
        layer_scale: Optional[float] = None,
        layer_scale_conv: Optional[float] = None,
        transformer_blocks: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if transformer_blocks is None:
            transformer_blocks = []

        self.transformer_block = not conv
        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(
                        dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale_conv,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=dim,
                        counter=i,
                        transformer_blocks=transformer_blocks,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        layer_scale=layer_scale,
                    )
                    for i in range(depth)
                ]
            )

        self.downsample = Downsample(dim=dim) if downsample else None
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
                _, _, hp, wp = x.shape
            else:
                hp, wp = h, w

            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)

        if self.transformer_block:
            x = window_reverse(x, self.window_size, hp, wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :h, :w].contiguous()

        if self.downsample is None:
            return x
        return self.downsample(x)


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
def _default_transformer_block_indices(depth: int) -> List[int]:
    """Match original block placement logic from the source implementation."""
    if depth % 2 != 0:
        return list(range(depth // 2 + 1, depth))
    return list(range(depth // 2, depth))


class MambaVision(nn.Module):
    """MambaVision image classification backbone + head."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        depths: Sequence[int],
        window_size: Sequence[int],
        mlp_ratio: float,
        num_heads: Sequence[int],
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        num_classes: int = 1000,
        qkv_bias: bool = True,
        qk_scale: Optional[bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        layer_scale: Optional[float] = None,
        layer_scale_conv: Optional[float] = None,
        **_: object,
    ) -> None:
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()

        for i, depth in enumerate(depths):
            conv_stage = i in (0, 1)
            transformer_blocks = _default_transformer_block_indices(depth)
            stage_drop_path = dpr[sum(depths[:i]) : sum(depths[: i + 1])]

            level = MambaVisionLayer(
                dim=int(dim * 2**i),
                depth=depth,
                num_heads=num_heads[i],
                window_size=window_size[i],
                conv=conv_stage,
                downsample=(i < 3),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_drop_path,
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks,
            )
            self.levels.append(level)

        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        return {"rpb"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)


__all__ = [
    "window_partition",
    "window_reverse",
    "Downsample",
    "PatchEmbed",
    "ConvBlock",
    "MambaVisionMixer",
    "Attention",
    "Block",
    "MambaVisionLayer",
    "MambaVision",
]
