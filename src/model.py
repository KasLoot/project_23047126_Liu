from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_dim = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.expand = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.reduce(scale), inplace=True)
        scale = torch.sigmoid(self.expand(scale))
        return x * scale


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: float,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(in_channels * expansion))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if hidden_dim != in_channels:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))

        layers.extend(
            [
                ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
                SqueezeExcite(hidden_dim),
                ConvBNAct(hidden_dim, out_channels, kernel_size=1, act=False),
            ]
        )

        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = x + self.drop_path(out)
        return out


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        expansion: float,
        drop_rates: list[float],
    ) -> None:
        super().__init__()
        blocks = [
            MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion=expansion,
                drop_path=drop_rates[0],
            )
        ]
        blocks.extend(
            MBConv(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                expansion=expansion,
                drop_path=drop_rates[index],
            )
            for index in range(1, depth)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class GestureEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, drop_path_rate: float = 0.08) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=2),
            ConvBNAct(32, 32, kernel_size=3, stride=1),
        )

        stage_defs = [
            (32, 64, 2, 2, 2.0),
            (64, 128, 2, 2, 2.0),
            (128, 192, 4, 2, 3.0),
            (192, 320, 3, 2, 3.0),
        ]
        total_blocks = sum(depth for _, _, depth, _, _ in stage_defs)
        drop_rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()

        stages = []
        cursor = 0
        for in_ch, out_ch, depth, stride, expansion in stage_defs:
            rates = drop_rates[cursor : cursor + depth]
            stages.append(EncoderStage(in_ch, out_ch, depth, stride, expansion, rates))
            cursor += depth

        self.stage1 = stages[0]
        self.stage2 = stages[1]
        self.stage3 = stages[2]
        self.stage4 = stages[3]
        self.out_channels = {"c2": 64, "c3": 128, "c4": 192, "c5": 320}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3, act=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)
        x = self.reduce(x)
        return self.act(x + self.refine(x))


class BiDirectionalPyramid(nn.Module):
    def __init__(self, backbone_channels: dict[str, int], pyramid_channels: int = 160) -> None:
        super().__init__()
        self.lateral2 = ConvBNAct(backbone_channels["c2"], pyramid_channels, kernel_size=1)
        self.lateral3 = ConvBNAct(backbone_channels["c3"], pyramid_channels, kernel_size=1)
        self.lateral4 = ConvBNAct(backbone_channels["c4"], pyramid_channels, kernel_size=1)
        self.lateral5 = ConvBNAct(backbone_channels["c5"], pyramid_channels, kernel_size=1)

        self.top4 = FusionBlock(pyramid_channels * 2, pyramid_channels)
        self.top3 = FusionBlock(pyramid_channels * 2, pyramid_channels)
        self.top2 = FusionBlock(pyramid_channels * 2, pyramid_channels)

        self.down_p2 = ConvBNAct(pyramid_channels, pyramid_channels, kernel_size=3, stride=2)
        self.down_p3 = ConvBNAct(pyramid_channels, pyramid_channels, kernel_size=3, stride=2)
        self.down_p4 = ConvBNAct(pyramid_channels, pyramid_channels, kernel_size=3, stride=2)

        self.out3 = FusionBlock(pyramid_channels * 2, pyramid_channels)
        self.out4 = FusionBlock(pyramid_channels * 2, pyramid_channels)
        self.out5 = FusionBlock(pyramid_channels * 2, pyramid_channels)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c2 = self.lateral2(features["c2"])
        c3 = self.lateral3(features["c3"])
        c4 = self.lateral4(features["c4"])
        c5 = self.lateral5(features["c5"])

        p5 = c5
        p4 = self.top4(c4, F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False))
        p3 = self.top3(c3, F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.top2(c2, F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False))

        n3 = self.out3(p3, self.down_p2(p2))
        n4 = self.out4(p4, self.down_p3(n3))
        n5 = self.out5(p5, self.down_p4(n4))

        return {"p2": p2, "p3": n3, "p4": n4, "p5": n5}


class SegmentationDecoder(nn.Module):
    def __init__(self, channels: int, out_channels: int = 1, decoder_channels: int = 128) -> None:
        super().__init__()
        self.project2 = ConvBNAct(channels, decoder_channels, kernel_size=1)
        self.project3 = ConvBNAct(channels, decoder_channels, kernel_size=1)
        self.project4 = ConvBNAct(channels, decoder_channels, kernel_size=1)
        self.project5 = ConvBNAct(channels, decoder_channels, kernel_size=1)

        self.decode4 = FusionBlock(decoder_channels * 2, decoder_channels)
        self.decode3 = FusionBlock(decoder_channels * 2, decoder_channels)
        self.decode2 = FusionBlock(decoder_channels * 2, decoder_channels)
        self.refine = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3),
        )
        self.mask_head = nn.Conv2d(decoder_channels, out_channels, kernel_size=1)
        self.attention_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p2 = self.project2(pyramid["p2"])
        p3 = self.project3(pyramid["p3"])
        p4 = self.project4(pyramid["p4"])
        p5 = self.project5(pyramid["p5"])

        d4 = self.decode4(p4, F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False))
        d3 = self.decode3(p3, F.interpolate(d4, size=p3.shape[-2:], mode="bilinear", align_corners=False))
        d2 = self.decode2(p2, F.interpolate(d3, size=p2.shape[-2:], mode="bilinear", align_corners=False))

        decoder_feat = self.refine(d2)
        seg_logits = self.mask_head(decoder_feat)
        seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
        attention_logits = self.attention_head(decoder_feat)
        return seg_logits, decoder_feat, attention_logits


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = x.shape
    ys = torch.linspace(-1.0, 1.0, steps=height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
    y_map = ys.expand(batch_size, 1, height, width)
    x_map = xs.expand(batch_size, 1, height, width)
    return torch.cat([x, x_map, y_map], dim=1)


def spatial_softmax(logits: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = logits.shape
    weights = logits.view(batch_size, 1, height * width)
    weights = weights.softmax(dim=-1)
    return weights.view(batch_size, 1, height, width)


def weighted_average_pool(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (x * weights).sum(dim=(2, 3))


def spatial_moments(weights: torch.Tensor) -> torch.Tensor:
    _, _, height, width = weights.shape
    ys = torch.linspace(0.0, 1.0, steps=height, device=weights.device, dtype=weights.dtype).view(1, 1, height, 1)
    xs = torch.linspace(0.0, 1.0, steps=width, device=weights.device, dtype=weights.dtype).view(1, 1, 1, width)

    mean_x = (weights * xs).sum(dim=(2, 3))
    mean_y = (weights * ys).sum(dim=(2, 3))

    var_x = (weights * (xs - mean_x.view(-1, 1, 1, 1)).pow(2)).sum(dim=(2, 3))
    var_y = (weights * (ys - mean_y.view(-1, 1, 1, 1)).pow(2)).sum(dim=(2, 3))

    std_x = torch.sqrt(var_x.clamp_min(1e-6))
    std_y = torch.sqrt(var_y.clamp_min(1e-6))
    peak = weights.amax(dim=(2, 3))
    return torch.cat([mean_x, mean_y, std_x, std_y, peak], dim=1)


class MultiScaleClassificationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, hidden_dim: int = 320, dropout: float = 0.25) -> None:
        super().__init__()
        self.project2 = ConvBNAct(channels, channels, kernel_size=1)
        self.project3 = ConvBNAct(channels, channels, kernel_size=1)
        self.project4 = ConvBNAct(channels, channels, kernel_size=1)
        self.project5 = ConvBNAct(channels, channels, kernel_size=1)
        self.project_decoder = ConvBNAct(128, channels, kernel_size=1)

        self.fuse = FusionBlock(channels * 5, channels)
        self.local_attention = nn.Conv2d(channels, 1, kernel_size=1)

        pooled_dim = channels * 4 + 1
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        decoder_feat: torch.Tensor,
        attention_logits: torch.Tensor,
    ) -> torch.Tensor:
        p2 = pyramid["p2"]
        target_size = p2.shape[-2:]

        fused = self.fuse(
            self.project2(p2),
            F.interpolate(self.project3(pyramid["p3"]), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.project4(pyramid["p4"]), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.project5(pyramid["p5"]), size=target_size, mode="bilinear", align_corners=False),
            self.project_decoder(decoder_feat),
        )

        att_logits = self.local_attention(fused) + attention_logits
        weights = spatial_softmax(att_logits)

        pooled_focus = weighted_average_pool(fused, weights)
        pooled_avg = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        pooled_max = F.adaptive_max_pool2d(fused, 1).flatten(1)
        context = torch.cat(
            [
                F.adaptive_avg_pool2d(pyramid["p5"], 1).flatten(1),
                F.adaptive_max_pool2d(pyramid["p5"], 1).flatten(1),
            ],
            dim=1,
        )
        gate_strength = torch.sigmoid(att_logits).mean(dim=(2, 3))

        embedding = torch.cat([pooled_focus, pooled_avg + pooled_max, context, gate_strength], dim=1)
        return self.classifier(embedding)


class SingleHandDetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, hidden_dim: int = 320, dropout: float = 0.2) -> None:
        super().__init__()
        self.project3 = ConvBNAct(channels, channels, kernel_size=1)
        self.project4 = ConvBNAct(channels, channels, kernel_size=1)
        self.project5 = ConvBNAct(channels, channels, kernel_size=1)
        self.project_decoder = ConvBNAct(128, channels, kernel_size=1)

        self.pre_fuse = FusionBlock(channels * 4, channels)
        self.coord_refine = nn.Sequential(
            ConvBNAct(channels + 2, channels, kernel_size=3),
            ConvBNAct(channels, channels, kernel_size=3),
        )
        self.attention_head = nn.Conv2d(channels, 1, kernel_size=1)

        pooled_dim = channels * 2 + 6
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        self.objectness_pred = nn.Linear(hidden_dim, 1)
        self.class_pred = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        decoder_feat: torch.Tensor,
        attention_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p3 = pyramid["p3"]
        target_size = p3.shape[-2:]

        fused = self.pre_fuse(
            self.project3(p3),
            F.interpolate(self.project4(pyramid["p4"]), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.project5(pyramid["p5"]), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.project_decoder(decoder_feat), size=target_size, mode="bilinear", align_corners=False),
        )
        fused = self.coord_refine(add_coord_channels(fused))

        att_logits = self.attention_head(fused) + F.interpolate(
            attention_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        weights = spatial_softmax(att_logits)

        pooled_focus = weighted_average_pool(fused, weights)
        pooled_global = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        geometry = spatial_moments(weights)
        att_strength = torch.sigmoid(att_logits).mean(dim=(2, 3))
        embedding = torch.cat([pooled_focus, pooled_global, geometry, att_strength], dim=1)

        shared = self.shared(embedding)
        return {
            "bbox": self.bbox_pred(shared),
            "objectness": self.objectness_pred(shared),
            "class_logits": self.class_pred(shared),
        }


class HandGestureMultiTask(nn.Module):
    def __init__(self, num_classes: int = 10, seg_out_channels: int = 1) -> None:
        super().__init__()
        self.backbone = GestureEncoder(in_channels=3, drop_path_rate=0.08)
        self.neck = BiDirectionalPyramid(self.backbone.out_channels, pyramid_channels=160)
        self.segmentation_head = SegmentationDecoder(channels=160, out_channels=seg_out_channels, decoder_channels=128)
        self.classification_head = MultiScaleClassificationHead(channels=160, num_classes=num_classes)
        self.detection_head = SingleHandDetectionHead(channels=160, num_classes=num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.constant_(self.detection_head.objectness_pred.bias, 2.0)
        with torch.no_grad():
            self.detection_head.bbox_pred.bias.copy_(torch.tensor([0.0, 0.0, -0.35, -0.35]))

    def forward(self, x: torch.Tensor, tasks: tuple[str, ...] | None = None) -> dict[str, object]:
        if tasks is None:
            tasks = ("classification", "detection", "segmentation")

        features = self.backbone(x)
        pyramid = self.neck(features)
        seg_logits, decoder_feat, attention_logits = self.segmentation_head(pyramid, output_size=x.shape[-2:])

        outputs: dict[str, object] = {}

        if "classification" in tasks or "cls" in tasks:
            outputs["cls"] = self.classification_head(pyramid, decoder_feat, attention_logits)

        if "detection" in tasks or "det" in tasks:
            outputs["det"] = self.detection_head(pyramid, decoder_feat, attention_logits)

        if "segmentation" in tasks or "seg" in tasks:
            outputs["seg"] = seg_logits

        return outputs
