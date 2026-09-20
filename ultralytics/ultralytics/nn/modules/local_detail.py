# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Local-detail modules for tiny-object YOLO variants.

These blocks intentionally avoid global attention. They bias the high-resolution
branches toward short-range texture, boundary contrast, and local spatial gating,
which is useful when the object is small and localization depends on a few pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import RepC2f
from .conv import Conv


class LocalDetailEnhancer(nn.Module):
    """Lightweight local detail enhancer with edge contrast and spatial gating."""

    def __init__(self, c: int, init_scale: float = 0.10):
        """Initialize local depthwise filters, pooling-edge branch, and residual scale.

        Args:
            c: Number of input/output channels.
            init_scale: Initial residual strength. Small values preserve pretrained
                YOLO behavior at the start of fine-tuning.
        """
        super().__init__()
        self.local3 = Conv(c, c, 3, g=c)
        self.local5 = Conv(c, c, 5, g=c)
        self.edge_conv = Conv(c, c, 3, g=c)
        self.mix = Conv(3 * c, c, 1)
        self.spatial_gate = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=True), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance local high-frequency details while preserving tensor shape."""
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge = x - smooth

        detail = self.mix(torch.cat((self.local3(x), self.local5(x), self.edge_conv(edge)), dim=1))
        gate = self.spatial_gate(torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1))
        return x + self.scale * detail * gate


class LocalDetailRepC2f(RepC2f):
    """RepC2f with branch-local detail enhancement for P2/P3 tiny-object features."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        init_scale: float = 0.10,
    ):
        """Initialize a RepC2f block with local detail enhancement on the active branch."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.local_detail = LocalDetailEnhancer(self.c, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local enhancement before the repeated Rep bottlenecks."""
        y = list(self.cv1(x).chunk(2, 1))
        y[1] = self.local_detail(y[1])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Split-based forward variant used by some export/fusion paths."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[1] = self.local_detail(y[1])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = ("LocalDetailEnhancer", "LocalDetailRepC2f")
