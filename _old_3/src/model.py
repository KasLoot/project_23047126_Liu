import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x
    
class Bottleneck(nn.Module):
    """Standard residual bottleneck."""

    def __init__(self, c1: int, c2: int, residual: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.residual else y
    

class C3k(nn.Module):
    """C3k block (stacked bottlenecks with configurable kernel-like behavior)."""

    def __init__(self, c: int, n: int = 2, residual: bool = True):
        super().__init__()
        self.m = nn.Sequential(*(Bottleneck(c, c, residual=residual, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class C3k2(nn.Module):
    """YOLO26 C3k2 block.

    For simplicity, this implementation uses the C2f scaffold with each repeated block
    being either Bottleneck or C3k.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        residual: bool = True,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, residual=residual, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

class SPPF(nn.Module):
    """Fast SPP block."""

    def __init__(self, c_in: int, c_out: int, k: int = 5, n: int = 3):
        super().__init__()
        c_ = c_in // 2
        self.cv1 = Conv(c_in, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        return self.cv2(torch.cat(y, 1))
    

class MHAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        assert self.d_k * num_head == d_model, "d_model must be divisible by num_head"
        self.qkv = nn.Conv2d(d_model, d_model * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.rms = nn.RMSNorm(d_model)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(batch_size, self.num_head, height * width, self.d_k)
        k = k.reshape(batch_size, self.num_head, height * width, self.d_k)
        v = v.reshape(batch_size, self.num_head, height * width, self.d_k)

        att = torch.einsum("bnqd,bnkd->bnqk", q, k) / math.sqrt(self.d_k)
        att = torch.softmax(att, dim=-1)
        att = torch.einsum("bnqk,bnqd->bnqd", att, v)
        output = self.out_proj(att.reshape(batch_size, self.d_model, height, width))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        self.attention = MHAttention(d_model=d_model, num_head=num_head)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.ffw1 = Conv(d_model, d_model * 4, kernel_size=1)
        self.ffw2 = Conv(d_model * 4, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        att = self.attention(x)
        x = self.norm1(x + self.dropout(att))
        ffw = self.ffw2(self.ffw1(x))
        x = x + self.dropout(ffw)
        return x


class CNNTransformer(nn.Module):
    def __init__(self, d_model, num_head, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.conv1 = Conv(d_model, d_model*2, kernel_size=3, stride=1)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(d_model, num_head=num_head, dropout=dropout) for _ in range(num_layers)
        )
        self.conv2 = Conv(d_model*2, d_model, kernel_size=3, stride=1)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.transformer_blocks:
            x_skip, x_process = torch.split(x, [self.d_model, self.d_model], dim=1)
            x_process = block(x_process)
            x = torch.cat([x_skip, x_process], dim=1)
            
        x = self.norm(self.conv2(x))
        return x
    

class ChannelMHAttention(nn.Module):
    """Linear-complexity channel-wise multi-head attention."""
    def __init__(self, d_model, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        assert self.d_k * num_head == d_model, "d_model must be divisible by num_head"
        
        self.qkv = nn.Conv2d(d_model, d_model * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        
        # Learnable temperature to scale the attention map
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # 1. Generate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 2. Reshape to (Batch, Heads, Channels_per_head, H*W)
        q = q.reshape(batch_size, self.num_head, self.d_k, height * width)
        k = k.reshape(batch_size, self.num_head, self.d_k, height * width)
        v = v.reshape(batch_size, self.num_head, self.d_k, height * width)

        # 3. L2 Normalize Q and K (Crucial for transposed attention stability)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 4. Compute Channel Attention: (B, Heads, C, HW) @ (B, Heads, HW, C) -> (B, Heads, C, C)
        # Notice we transpose the HW dimension, NOT the channels
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 5. Apply attention to V: (B, Heads, C, C) @ (B, Heads, C, HW) -> (B, Heads, C, HW)
        out = (attn @ v)
        
        # 6. Reshape and project
        out = out.reshape(batch_size, self.d_model, height, width)
        return self.out_proj(out)


class ChannelTransformerBlock(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        # Swap in the new efficient attention
        self.attention = ChannelMHAttention(d_model=d_model, num_head=num_head)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        
        # Keep FFN as 1x1 convs
        self.ffw1 = nn.Conv2d(d_model, d_model * 4, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.ffw2 = nn.Conv2d(d_model * 4, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture is generally more stable for deep transformers
        att = self.attention(self.norm1(x))
        x = x + self.dropout(att)
        
        ffw = self.ffw2(self.act(self.ffw1(self.norm2(x))))
        x = x + self.dropout(ffw)
        return x


class ChannelTransformer(nn.Module):
    def __init__(self, d_model, num_head, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # CSP-style embedding
        self.conv1 = nn.Conv2d(d_model, d_model * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_in = nn.BatchNorm2d(d_model * 2)
        self.act_in = nn.SiLU(inplace=True)
        
        self.transformer_blocks = nn.ModuleList(
            ChannelTransformerBlock(d_model, num_head=num_head, dropout=dropout) for _ in range(num_layers)
        )
        
        self.conv2 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_out = nn.BatchNorm2d(d_model)
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act_in(self.norm_in(self.conv1(x)))
        
        # CSP (Cross Stage Partial) split
        x_skip, x_process = torch.split(x, [self.d_model, self.d_model], dim=1)
        
        for block in self.transformer_blocks:
            x_process = block(x_process)
            
        x = torch.cat([x_skip, x_process], dim=1)
        x = self.act_out(self.norm_out(self.conv2(x)))
        return x


class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=self.dim)


class ModelConfig:
    feature_channels: list[int] = [16, 32, 64, 128, 256]
    num_heads: int = feature_channels[-1] // 64
    num_classes: int = 10

class BackBone_v1(nn.Module):
    def __init__(self, model_config: ModelConfig | None =None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        
        self.f1_1 = Conv(3, self.model_config.feature_channels[0], kernel_size=3, stride=2)
        self.f1_2 = Conv(self.model_config.feature_channels[0], self.model_config.feature_channels[1], kernel_size=3, stride=2)
        self.f1_3 = C3k2(self.model_config.feature_channels[1], self.model_config.feature_channels[2], n=2)

        self.f2_1 = Conv(self.model_config.feature_channels[2], self.model_config.feature_channels[2], kernel_size=3, stride=2)
        self.f2_2 = C3k2(self.model_config.feature_channels[2], self.model_config.feature_channels[3], n=2)

        self.f3_1 = Conv(self.model_config.feature_channels[3], self.model_config.feature_channels[3], kernel_size=3, stride=2)
        self.f3_2 = C3k2(self.model_config.feature_channels[3], self.model_config.feature_channels[3], n=2)

        self.f4_1 = Conv(self.model_config.feature_channels[3], self.model_config.feature_channels[4], kernel_size=3, stride=2)
        self.f4_2 = CNNTransformer(d_model=self.model_config.feature_channels[4], num_head=self.model_config.num_heads, num_layers=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.cat = Concat(dim=1)

        self.n1 = C3k2(self.model_config.feature_channels[4] + self.model_config.feature_channels[3], self.model_config.feature_channels[3], n=2)
        self.n2 = C3k2(self.model_config.feature_channels[3] + self.model_config.feature_channels[3], self.model_config.feature_channels[2], n=2)
        self.n3 = C3k2(self.model_config.feature_channels[2] + self.model_config.feature_channels[2], self.model_config.feature_channels[1], n=2)
        self.n4 = C3k2(self.model_config.feature_channels[1] + self.model_config.feature_channels[0], self.model_config.feature_channels[0], n=2)


    def forward(self, x):
        f0 = self.f1_1(x)
        f1 = self.f1_3(self.f1_2(f0))
        f2 = self.f2_2(self.f2_1(f1))
        f3 = self.f3_2(self.f3_1(f2))
        f4 = self.f4_2(self.f4_1(f3))

        n1 = self.n1(self.cat([self.up(f4), f3]))
        n2 = self.n2(self.cat([self.up(n1), f2]))
        n3 = self.n3(self.cat([self.up(n2), f1]))
        n4 = self.n4(self.cat([self.up(n3), f0]))

        # ADDED f4 to the outputs so it can be passed to the DetectionHead (Stride 32)
        return [f4, n1, n2, n3, n4]
    

class ClassifierHead_v1(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        self.cls_head = nn.Sequential(
            Conv(self.model_config.feature_channels[-2], self.model_config.feature_channels[-2] * 2, kernel_size=3, stride=1),
            Conv(self.model_config.feature_channels[-2] * 2, self.model_config.feature_channels[-2] * 2, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model_config.feature_channels[-2] * 2, self.model_config.feature_channels[-2] * 2),
            nn.RMSNorm(self.model_config.feature_channels[-2] * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.model_config.feature_channels[-2] * 2, self.model_config.num_classes)
        )

    def forward(self, x):
        logits = self.cls_head(x)
        return logits


class DetectionHead_v1(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
            
        # Select target channels for Strides 32, 16, and 8
        det_channels = [
            self.model_config.feature_channels[4], # 256 (f4)
            self.model_config.feature_channels[3], # 128 (n1)
            self.model_config.feature_channels[2]  # 64  (n2)
        ]

        self.box_head = nn.ModuleList([
            nn.Sequential(
                Conv(ch, self.model_config.feature_channels[1], kernel_size=3, stride=1),
                Conv(self.model_config.feature_channels[1], self.model_config.feature_channels[1], kernel_size=3, stride=1),
                nn.Conv2d(self.model_config.feature_channels[1], 4, kernel_size=1)
            )
            for ch in det_channels
        ])

        self.cls_head = nn.ModuleList([
            nn.Sequential(
                Conv(ch, self.model_config.feature_channels[1], kernel_size=3, stride=1),
                Conv(self.model_config.feature_channels[1], self.model_config.feature_channels[1], kernel_size=3, stride=1),
                nn.Conv2d(self.model_config.feature_channels[1], self.model_config.num_classes, kernel_size=1)
            )
            for ch in det_channels
        ])

        self._initialize_biases()

    def _initialize_biases(self):
        prior = 0.01
        for m in self.cls_head:
            last_conv = m[-1]
            nn.init.constant_(last_conv.bias, -math.log((1 - prior) / prior))

    def forward(self, x: list[torch.Tensor]):
        bboxes = torch.cat([head(x[i]).view(x[i].shape[0], 4, -1) for i, head in enumerate(self.box_head)], dim=-1)
        bbox_cls = torch.cat([head(x[i]).view(x[i].shape[0], self.model_config.num_classes, -1) for i, head in enumerate(self.cls_head)], dim=-1)
        return bboxes, bbox_cls


class SegmentationHead_v1(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        self.seg_head = nn.Sequential(
            Conv(sum(self.model_config.feature_channels[:-1]), self.model_config.feature_channels[-1], kernel_size=3, stride=1),
            Conv(self.model_config.feature_channels[-1], self.model_config.feature_channels[-1], kernel_size=3, stride=1),
            nn.Conv2d(self.model_config.feature_channels[-1], 1, kernel_size=1)
        )

    def forward(self, x):
        # x is now [n1, n2, n3, n4]
        target_size = x[-1].shape[2:]
        n1_up = F.interpolate(x[0], size=target_size, mode='nearest')
        n2_up = F.interpolate(x[1], size=target_size, mode='nearest')
        n3_up = F.interpolate(x[2], size=target_size, mode='nearest')
        seg_map = self.seg_head(torch.cat([n1_up, n2_up, n3_up, x[-1]], dim=1))
        seg_map = F.interpolate(seg_map, scale_factor=2, mode='nearest') 
        return seg_map


class HandGestureModel_v1(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        self.backbone = BackBone_v1(model_config=self.model_config)
        self.classifier_head = ClassifierHead_v1(model_config=self.model_config)
        self.detection_head = DetectionHead_v1(model_config=self.model_config)
        self.segmentation_head = SegmentationHead_v1(model_config=self.model_config)

    def forward(self, x):
        features = self.backbone(x) # [f4, n1, n2, n3, n4]
        print(f"Backbone_v1 output shapes:\n f4: {features[0].shape}\n n1: {features[1].shape}\n n2: {features[2].shape}\n n3: {features[3].shape}\n n4: {features[4].shape}")
        
        # Route specifically to avoid passing the massive n3 and n4 feature maps to detection
        cls_logits = self.classifier_head(features[1])            # Use n1 (128 channels)
        bbox_preds, bbox_cls = self.detection_head(features[:3])  # Use f4, n1, n2
        seg_map = self.segmentation_head(features[1:])            # Use n1, n2, n3, n4

        # print(f"bbox_preds shape: {bbox_preds.shape}, bbox_cls shape: {bbox_cls.shape}, seg_map shape: {seg_map.shape}")
        
        return cls_logits, bbox_preds, bbox_cls, seg_map


class BackBone_v2(nn.Module):
    def __init__(self, model_config: ModelConfig | None =None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        
        self.f1_1 = Conv(3, self.model_config.feature_channels[0], kernel_size=3, stride=2)
        self.f1_2 = Conv(self.model_config.feature_channels[0], self.model_config.feature_channels[1], kernel_size=3, stride=2)
        self.f1_3 = C3k2(self.model_config.feature_channels[1], self.model_config.feature_channels[2], n=2)

        self.f2_1 = Conv(self.model_config.feature_channels[2], self.model_config.feature_channels[2], kernel_size=3, stride=2)
        self.f2_2 = C3k2(self.model_config.feature_channels[2], self.model_config.feature_channels[3], n=2)

        self.f3_1 = Conv(self.model_config.feature_channels[3], self.model_config.feature_channels[3], kernel_size=3, stride=2)
        self.f3_2 = C3k2(self.model_config.feature_channels[3], self.model_config.feature_channels[3], n=2)

        self.f4_1 = Conv(self.model_config.feature_channels[3], self.model_config.feature_channels[4], kernel_size=3, stride=2)
        self.f4_2 = ChannelTransformer(d_model=self.model_config.feature_channels[4], num_head=self.model_config.num_heads, num_layers=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.cat = Concat(dim=1)

        self.n1 = C3k2(self.model_config.feature_channels[4] + self.model_config.feature_channels[3], self.model_config.feature_channels[3], n=2)
        self.n2 = C3k2(self.model_config.feature_channels[3] + self.model_config.feature_channels[3], self.model_config.feature_channels[2], n=2)
        self.n3 = C3k2(self.model_config.feature_channels[2] + self.model_config.feature_channels[2], self.model_config.feature_channels[1], n=2)
        self.n4 = C3k2(self.model_config.feature_channels[1] + self.model_config.feature_channels[0], self.model_config.feature_channels[0], n=2)


    def forward(self, x):
        f0 = self.f1_1(x)
        f1 = self.f1_3(self.f1_2(f0))
        f2 = self.f2_2(self.f2_1(f1))
        f3 = self.f3_2(self.f3_1(f2))
        f4 = self.f4_2(self.f4_1(f3))

        n1 = self.n1(self.cat([self.up(f4), f3]))
        n2 = self.n2(self.cat([self.up(n1), f2]))
        n3 = self.n3(self.cat([self.up(n2), f1]))
        n4 = self.n4(self.cat([self.up(n3), f0]))

        # ADDED f4 to the outputs so it can be passed to the DetectionHead (Stride 32)
        return [f4, n1, n2, n3, n4]





class BackBone_v3(nn.Module):
    def __init__(self, model_config: ModelConfig | None =None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
            
        c0, c1, c2, c3, c4 = self.model_config.feature_channels
        
        # P1/2
        self.b0 = Conv(3, c0, kernel_size=3, stride=2)
        
        # P2/4
        self.b1 = Conv(c0, c1, kernel_size=3, stride=2)
        self.b2 = C3k2(c1, c2, n=2)
        
        # P3/8
        self.b3 = Conv(c2, c2, kernel_size=3, stride=2)
        self.b4 = C3k2(c2, c2, n=2)
        
        # P4/16
        self.b5 = Conv(c2, c3, kernel_size=3, stride=2)
        self.b6 = C3k2(c3, c3, n=2)
        
        # P5/32
        self.b7 = Conv(c3, c4, kernel_size=3, stride=2)
        self.b8 = ChannelTransformer(d_model=c4, num_head=self.model_config.num_heads, num_layers=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.cat = Concat(dim=1)

        # Neck (FPN + PAN)
        # Up-path (P5 -> P4 -> P3)
        self.up_p4 = C3k2(c4 + c3, c3, n=2)
        self.up_p3 = C3k2(c3 + c2, c2, n=2)

        # Down-path (P3 -> P4 -> P5)
        self.down_p4 = Conv(c2, c2, kernel_size=3, stride=2)
        self.pan_p4 = C3k2(c2 + c3, c3, n=2)
        
        self.down_p5 = Conv(c3, c3, kernel_size=3, stride=2)
        self.pan_p5 = C3k2(c3 + c4, c4, n=2)

    def forward(self, x):
        # Backbone
        x = self.b0(x)       # 1/2
        x = self.b2(self.b1(x)) # 1/4 (P2)
        
        p3 = self.b4(self.b3(x)) # 1/8, c2
        p4 = self.b6(self.b5(p3)) # 1/16, c3
        p5 = self.b8(self.b7(p4)) # 1/32, c4

        # Neck (PAN)
        # Top-down
        p4_up = self.up_p4(self.cat([self.up(p5), p4]))
        p3_out = self.up_p3(self.cat([self.up(p4_up), p3]))

        # Bottom-up
        p4_out = self.pan_p4(self.cat([self.down_p4(p3_out), p4_up]))
        p5_out = self.pan_p5(self.cat([self.down_p5(p4_out), p5]))

        # Return multi-scale features for Detection and Segmentation
        return [p3_out, p4_out, p5_out]  # (1/8, 1/16, 1/32)

class ClassifierHead_v3(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
            
        c5 = self.model_config.feature_channels[4]
        self.cls_head = nn.Sequential(
            Conv(c5, c5, kernel_size=3, stride=2), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c5, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.model_config.num_classes)
        )

    def forward(self, x):
        logits = self.cls_head(x)
        return logits

class DetectionHead_v3(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
            
        # Target channels for Strides 8, 16, 32
        det_channels = [
            self.model_config.feature_channels[2], # 64
            self.model_config.feature_channels[3], # 128
            self.model_config.feature_channels[4]  # 256
        ]

        c2 = max(16, det_channels[0] // 4)
        c3 = max(det_channels[0], self.model_config.num_classes)

        self.box_head = nn.ModuleList([
            nn.Sequential(
                Conv(ch, c2, kernel_size=3, stride=1),
                Conv(c2, c2, kernel_size=3, stride=1),
                nn.Conv2d(c2, 4, kernel_size=1)
            )
            for ch in det_channels
        ])

        self.cls_head = nn.ModuleList([
            nn.Sequential(
                Conv(ch, c3, kernel_size=3, stride=1),
                Conv(c3, c3, kernel_size=3, stride=1),
                nn.Conv2d(c3, self.model_config.num_classes, kernel_size=1)
            )
            for ch in det_channels
        ])

        self._initialize_biases()

    def _initialize_biases(self):
        prior = 0.01
        for m in self.cls_head:
            last_conv = m[-1]
            nn.init.constant_(last_conv.bias, -math.log((1 - prior) / prior))

    def forward(self, x: list[torch.Tensor]):
        bboxes = torch.cat([head(x[i]).view(x[i].shape[0], 4, -1) for i, head in enumerate(self.box_head)], dim=-1)
        bbox_cls = torch.cat([head(x[i]).view(x[i].shape[0], self.model_config.num_classes, -1) for i, head in enumerate(self.cls_head)], dim=-1)
        return bboxes, bbox_cls

class SegmentationHead_v3(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
            
        c_in = self.model_config.feature_channels[2] + self.model_config.feature_channels[3] + self.model_config.feature_channels[4] # 64 + 128 + 256 = 448
        mid_c = 128 
        
        self.seg_conv = nn.Sequential(
            Conv(c_in, mid_c, kernel_size=3, stride=1),
            Conv(mid_c, mid_c, kernel_size=3, stride=1),
            nn.Conv2d(mid_c, 1, kernel_size=1)
        )

    def forward(self, x):
        # x is [p3, p4, p5] -> strides (1/8, 1/16, 1/32)
        p3, p4, p5 = x
        target_size = p3.shape[2:]
        p4_up = F.interpolate(p4, size=target_size, mode='nearest')
        p5_up = F.interpolate(p5, size=target_size, mode='nearest')
        
        fused = torch.cat([p3, p4_up, p5_up], dim=1)
        mask_low_res = self.seg_conv(fused)
        
        # upsample back to original (8x)
        seg_map = F.interpolate(mask_low_res, scale_factor=8, mode='bilinear', align_corners=False)
        return seg_map

class HandGestureModel_v3(nn.Module):
    def __init__(self, model_config: ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            self.model_config = ModelConfig()
        else:
            self.model_config = model_config
        self.backbone = BackBone_v3(model_config=self.model_config)
        self.classifier_head = ClassifierHead_v3(model_config=self.model_config)
        self.detection_head = DetectionHead_v3(model_config=self.model_config)
        self.segmentation_head = SegmentationHead_v3(model_config=self.model_config)

    def forward(self, x):
        features = self.backbone(x) # [p3, p4, p5]
        
        cls_logits = self.classifier_head(features[-1])            
        bbox_preds, bbox_cls = self.detection_head(features)  
        seg_map = self.segmentation_head(features)            
        
        return cls_logits, bbox_preds, bbox_cls, seg_map



if __name__ == "__main__":
    # Test suite adapted to the new returned feature list outputs
    model = C3k2(3, 128, n=2)
    x = torch.randn(1, 3, 640, 480)
    output = model(x)
    print(f"final output.shape: {output.shape}")

    print("\nTesting SPPF block:")
    model = SPPF(3, 128, k=5, n=3)
    x = torch.randn(1, 3, 640, 480)
    output = model(x)
    print(f"SPPF output.shape: {output.shape}")

    print("\nTesting MHAttention block:")
    model = MHAttention(d_model=128, num_head=4)
    x = torch.randn(1, 128, 64, 64)
    output = model(x)
    print(f"MHAttention output.shape: {output.shape}")

    print("\nTesting TransformerBlock:")
    model = TransformerBlock(d_model=128, num_head=4)
    x = torch.randn(1, 128, 64, 64)
    output = model(x)
    print(f"TransformerBlock output.shape: {output.shape}")

    print("\nTesting CNNTransformer:")
    model = CNNTransformer(d_model=128, num_head=4, num_layers=2)
    x = torch.randn(1, 128, 64, 64)
    output = model(x)
    print(f"CNNTransformer output.shape: {output.shape}")

    print("\nTesting BackBone:")
    model = BackBone_v1()
    x = torch.randn(1, 3, 640, 480)
    backbone_output = model(x)
    print(f"BackBone output shapes: {[out.shape for out in backbone_output]}")

    print("\nTesting ClassifierHead:")
    model = ClassifierHead_v1()
    # Expects features[1] (n1)
    output = model(backbone_output[1])
    print(f"ClassifierHead output.shape: {output.shape}")

    print("\nTesting DetectionHead:")
    model = DetectionHead_v1()
    bbox, bbox_cls = model(backbone_output[:3])
    print(f"DetectionHead bbox output.shape: {bbox.shape}, bbox_cls output.shape: {bbox_cls.shape}")

    print("\nTesting SegmentationHead:")
    model = SegmentationHead_v1()
    seg_map = model(backbone_output[1:])
    print(f"SegmentationHead output.shape: {seg_map.shape}")

    print("\nTesting HandGestureModel:")
    model = HandGestureModel_v1()
    model_input = torch.randn(1, 3, 640, 480)
    cls_logits, bbox_preds, bbox_cls, seg_map = model(model_input)
    print(f"HandGestureModel cls_logits shape: {cls_logits.shape}, bbox_preds shape: {bbox_preds.shape}, bbox_cls shape: {bbox_cls.shape}, seg_map shape: {seg_map.shape}")

    import torchinfo
    torchinfo.summary(model, input_size=(1, 3, 640, 480))





    print("\nTesting BackBone_v2:")
    backbone_v2 = BackBone_v2()
    x = torch.randn(1, 3, 640, 480)
    backbone_v2_output = backbone_v2(x)
    print(f"BackBone_v2 output shapes: {[out.shape for out in backbone_v2_output]}")
    torchinfo.summary(backbone_v2, input_size=(1, 3, 640, 480))
    print('\nTesting BackBone_v3:')
    backbone_v3 = BackBone_v3()
    backbone_v3_output = backbone_v3(x)
    print(f'BackBone_v3 output shapes: {[out.shape for out in backbone_v3_output]}')
    model_v3 = HandGestureModel_v3()
    model_v3.eval()
    torchinfo.summary(model_v3, input_size=(1, 3, 640, 480))
