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
        print(f"2*c: {2 * self.c}, (2 + n)*c: {(2 + n) * self.c}")
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, residual=residual, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        print(f"After chunk: y[0] shape: {y[0].shape}, y[1] shape: {y[1].shape}")
        print(f"y[-1] shape before loop: {y[-1].shape}")
        y.extend(m(y[-1]) for m in self.m)
        print(f"y length: {len(y)}, shapes: {[t.shape for t in y]}")
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


if __name__ == "__main__":
    model = C3k2(3, 128, n=2)
    x = torch.randn(1, 3, 640, 480)
    output = model(x)
    print(f"final output.shape: {output.shape}")

    print("\nTesting SPPF block:")

    model = SPPF(3, 128, k=5, n=3)
    x = torch.randn(1, 3, 640, 480)
    output = model(x)
    print(f"SPPF output.shape: {output.shape}")
