import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def autopad(k: int, p: int | None = None):
    return k // 2 if p is None else p

class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation."""
    def __init__(self, ci, co, k=1, s=1, p=None, act=True, g=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, autopad(k, p), groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ci, co, residual=True):
        super().__init__()
        self.conv1 = Conv(ci, co, 3, 1)
        self.conv2 = Conv(co, co, 3, 1)
        self.residual = residual

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class CSP(nn.Module):
    """
    Cross Stage Partial (CSP) block.

    Idea:
      1) Split features into two paths.
      2) Process only one path through bottlenecks.
      3) Concatenate processed + shortcut path.
      4) Fuse with a final 1x1 convolution.

    This reduces compute/memory while preserving gradient flow.
    """
    def __init__(self, ci, co, n=1, e=0.5, residual=True):
        super().__init__()
        hidden = int(co * e)

        # Split
        self.cv1 = Conv(ci, hidden, k=1, s=1)  # processed branch
        self.cv2 = Conv(ci, hidden, k=1, s=1)  # shortcut branch

        # Transform processed branch
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, residual=residual) for _ in range(n)])

        # Fuse
        self.cv3 = Conv(2 * hidden, co, k=1, s=1)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPP(nn.Module):
    """Spatial Pyramid Pooling block using parallel max-pooling kernels."""
    def __init__(self, ci, co, k=(5, 9, 13)):
        super().__init__()
        hidden = max(1, ci // 2)
        self.cv1 = Conv(ci, hidden, k=1, s=1)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in k
        ])
        self.cv2 = Conv(hidden * (len(k) + 1), co, k=1, s=1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [pool(x) for pool in self.m], dim=1))
    

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
    

class PSA(nn.Module):
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
        self.m = nn.Sequential(*(PSA(self.c) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))
    


class ClassifierHead(nn.Module):
    def __init__(self, ci, num_classes):
        super().__init__()
        self.conv1 = Conv(ci, ci, k=3, s=1)
        self.conv2 = Conv(ci, ci, k=2)
        self.conv3 = Conv(ci, ci, k=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ffw1 = nn.Linear(ci, ci)
        self.ffw2 = nn.Linear(ci, ci)
        self.out = nn.Linear(ci, num_classes)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.pool(x).flatten(1)
        x = F.dropout(F.silu(F.rms_norm(self.ffw1(x))), p=0.3, training=self.training)
        x = F.dropout(F.rms_norm(self.ffw2(x)), p=0.3, training=self.training)
        return self.out(x)
    

class SegmentationHead(nn.Module):
    def __init__(self, ci, num_classes):
        super().__init__()
        self.conv1 = Conv(ci, ci, k=3, s=1)
        self.conv2 = Conv(ci, ci, k=2)
        self.conv3 = Conv(ci, ci, k=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ffw1 = nn.Linear(ci, ci)
        self.ffw2 = nn.Linear(ci, ci)
        self.out = nn.Linear(ci, num_classes)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.pool(x).flatten(1)
        x = F.dropout(F.silu(F.rms_norm(self.ffw1(x))), p=0.3, training=self.training)
        x = F.dropout(F.rms_norm(self.ffw2(x)), p=0.3, training=self.training)
        return self.out(x)


class DetectionHead(nn.Module):
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

        if self.training:
            if self.end2end:
                x_detach = [xi.detach() for xi in x]
                one2one = self._forward_branch(x_detach, self.one2one_cv2, self.one2one_cv3)
                return {"one2many": one2many, "one2one": one2one}
            return one2many

        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self._forward_branch(x_detach, self.one2one_cv2, self.one2one_cv3)
            y = self._postprocess_one2one(one2one["boxes"], one2one["scores"])
            return y, {"one2many": one2many, "one2one": one2one}

        return one2many
    
class Concat(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=self.dim)

class Core(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128, 256)):
        super().__init__()

        # Backbone
        self.b0 = Conv(3, ch[0], 3, 2)
        self.b1 = Conv(ch[0], ch[1], 3, 2)
        self.b2 = CSP(ch[1], ch[2], n=3)
        self.b3 = CSP(ch[2], ch[3], n=9)
        self.b4 = SPP(ch[3], ch[4])
        self.b5 = C2PSA(ch[4], ch[4])

        # Neck
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cat = Concat(1)

        self.n1_1 = Conv(ch[1]+ch[2], ch[2], 3, 1)
        self.n1_2 = CSP(ch[2], ch[2], n=3, e=0.5)

        self.n2_1 = Conv(ch[2]+ch[3], ch[3], 3, 1)
        self.n2_2 = CSP(ch[3], ch[3], n=3, e=0.5)

        self.n3_1 = Conv(ch[3]+ch[4], ch[4], 3, 1)
        self.n3_2 = CSP(ch[4], ch[4], n=3, e=0.5)

    def forward(self, x):
        # Backbone
        x0 = self.b1(self.b0(x)) #128
        x1 = self.b2(x0) #256
        x2 = self.b3(x1) #512
        x3 = self.b5(self.b4(x2)) #1024

        print(x0.shape, x1.shape, x2.shape, x3.shape)


        




if __name__ == "__main__":
    import torchinfo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Core().to(torch.float16).to(device)
    torchinfo.summary(model, input_size=(1, 3, 640, 480), dtypes=[torch.float16], device=device, depth=5)
    model.eval()

    x = torch.randn(1, 3, 640, 480).to(torch.float16).to(device)
    y = model(x)