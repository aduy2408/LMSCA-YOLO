"""Lightweight RGB-derived color cue fusion for the next FTSC ablation.

This is a project adaptation motivated by BDNet's color-opponent/hue pathway. It
keeps the pretrained three-channel RGB backbone untouched and fuses a zero
initialized residual at P2. It is intentionally not a full BDNet implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("P2ColorCueFusion", "rgb_color_cues")


def rgb_color_cues(image: torch.Tensor) -> torch.Tensor:
    """Return finite ``[R-G, (R+G)/2-B, sin(H), cos(H)]`` cues from RGB in [0, 1]."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB input, got {tuple(image.shape)}")
    rgb = image.float().clamp(0.0, 1.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maximum = rgb.max(dim=1).values
    minimum = rgb.min(dim=1).values
    delta = maximum - minimum
    safe = delta.clamp_min(torch.finfo(rgb.dtype).eps)
    # Standard hue sectors in turns. The branch is only used where chroma exists;
    # achromatic pixels are assigned hue zero and therefore (sin, cos)=(0, 1).
    h = torch.zeros_like(maximum)
    rmask = (maximum == r) & (delta > 0)
    gmask = (maximum == g) & (delta > 0)
    bmask = (maximum == b) & (delta > 0)
    h = torch.where(rmask, ((g - b) / safe).remainder(6.0), h)
    h = torch.where(gmask, ((b - r) / safe) + 2.0, h)
    h = torch.where(bmask, ((r - g) / safe) + 4.0, h)
    h = h.remainder(6.0) * (torch.pi / 3.0)
    cues = torch.stack((r - g, 0.5 * (r + g) - b, torch.sin(h), torch.cos(h)), dim=1)
    return cues.nan_to_num(0.0, 0.0, 0.0)


class P2ColorCueFusion(nn.Module):
    """RGB-derived cue encoder with identity-friendly residual P2 fusion.

    ``forward`` receives the P2 feature and the original (already augmented)
    RGB image. The final 1x1 projection is zero initialized, so a freshly
    constructed module is exactly the RGB reference function.
    """

    needs_image = True
    cue_channels = 4

    def __init__(self, p2_channels: int, hidden: int = 32):
        super().__init__()
        hidden = max(8, int(hidden))
        self.encoder = nn.Sequential(
            Conv(self.cue_channels, hidden, 3, 2),
            Conv(hidden, hidden, 3, 2),
            Conv(hidden, hidden, 3, 1),
        )
        self.projection = nn.Conv2d(hidden, p2_channels, 1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.last_stats: dict[str, torch.Tensor] = {}

    def estimated_gflops(self, image_size: int = 512) -> float:
        """Static multiply-add estimate for the cue branch at a square input."""
        h = image_size
        total = 0
        for cin, cout, k, stride in ((4, self.encoder[0].conv.out_channels, 3, 2), (self.encoder[0].conv.out_channels, self.encoder[1].conv.out_channels, 3, 2), (self.encoder[1].conv.out_channels, self.encoder[2].conv.out_channels, 3, 1)):
            h = (h + 2 * (k // 2) - k) // stride + 1
            total += 2 * cin * cout * k * k * h * h
        total += 2 * self.projection.in_channels * self.projection.out_channels * h * h
        return total / 1e9

    def forward(self, p2: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        cues = rgb_color_cues(image).to(device=p2.device, dtype=p2.dtype)
        feature = self.encoder(cues)
        if feature.shape[-2:] != p2.shape[-2:]:
            feature = torch.nn.functional.interpolate(feature, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.projection(feature)
        rgb_norm = p2.detach().flatten(1).norm(dim=1).mean().clamp_min(1e-8)
        color_norm = residual.detach().flatten(1).norm(dim=1).mean()
        self.last_stats = {
            "rgb_p2_norm": rgb_norm,
            "color_residual_norm": color_norm,
            "color_rgb_norm_ratio": color_norm / rgb_norm,
            "color_rgb_cosine": torch.nn.functional.cosine_similarity(residual.detach().flatten(1), p2.detach().flatten(1), dim=1).mean(),
            "projection_weight_norm": self.projection.weight.detach().norm(),
            "opponent_mean": cues[:, :2].detach().mean(),
            "opponent_std": cues[:, :2].detach().std(unbiased=False),
            "hue_sin_mean": cues[:, 2].detach().mean(),
            "hue_cos_mean": cues[:, 3].detach().mean(),
        }
        return p2 + residual
