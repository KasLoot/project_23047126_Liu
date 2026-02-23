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




a = torch.randn(1, 3, 640, 480)
model = Conv(3, 16, 3, 1)
print(model(a).shape)  # Should print torch.Size([1, 16, 640, 480])
