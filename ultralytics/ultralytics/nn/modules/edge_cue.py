"""Lightweight orientation-selective edge cue fusion.

This is a BDNet-inspired project adaptation, not the full BDNet OrSM. Fixed
oriented filters preserve directional information; a learned softmax gate selects
orientations before a zero-initialized P2 residual projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("P2EdgeCueFusion", "oriented_edge_responses")


def _oriented_kernels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Derivative-like 3x3 filters for 0, 45, 90, 135 degrees.
    kernels = torch.tensor(
        [
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        ],
        dtype=dtype,
    )
    return kernels / kernels.flatten(1).norm(dim=1).view(4, 1, 1).clamp_min(1e-8)


def oriented_edge_responses(image: torch.Tensor) -> torch.Tensor:
    """Return four finite oriented luminance responses with shape ``B,4,H,W``."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB input, got {tuple(image.shape)}")
    rgb = image.float().clamp(0.0, 1.0)
    gray = (0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    kernels = _oriented_kernels(gray.dtype).to(gray.device).unsqueeze(1)
    return F.conv2d(gray, kernels, padding=1).nan_to_num(0.0, 0.0, 0.0)


class P2EdgeCueFusion(nn.Module):
    """Orientation-selective edge branch with identity-friendly P2 fusion."""

    needs_image = True
    orientation_count = 4

    def __init__(
        self,
        p2_channels: int,
        hidden: int = 32,
        residual_scale: float = 1.0,
        residual_schedule: str = "constant",
        ramp_start_epoch: int = 5,
        ramp_end_epoch: int = 15,
        orientation_gate_mode: str = "learned",
    ):
        super().__init__()
        hidden = max(8, int(hidden))
        self.hidden = hidden
        self.residual_scale = float(residual_scale)
        self.residual_schedule = str(residual_schedule).lower()
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.orientation_gate_mode = str(orientation_gate_mode).lower()
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if self.residual_schedule not in {"constant", "ramp"}:
            raise ValueError("residual_schedule must be 'constant' or 'ramp'")
        if self.residual_schedule == "ramp" and self.ramp_end_epoch <= self.ramp_start_epoch:
            raise ValueError("ramp_end_epoch must be greater than ramp_start_epoch")
        if self.orientation_gate_mode not in {"learned", "uniform"}:
            raise ValueError("orientation_gate_mode must be 'learned' or 'uniform'")
        self.encoder = nn.Sequential(
            Conv(self.orientation_count, hidden, 3, 2),
            Conv(hidden, hidden, 3, 2),
            Conv(hidden, hidden, 3, 1),
        )
        self.gate = (
            nn.Linear(self.orientation_count, self.orientation_count)
            if self.orientation_gate_mode == "learned"
            else None
        )
        self.projection = nn.Conv2d(hidden, p2_channels, 1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.register_buffer("orientation_kernels", _oriented_kernels(), persistent=False)
        # ``last_stats`` is intentionally only a compatibility view.  The
        # committed phase-specific snapshots below prevent profile, warmup,
        # export, and validation forwards from replacing the training sample
        # consumed by the mechanism logger.
        self.last_stats: dict[str, torch.Tensor] = {}
        self._committed_stats: dict[str, torch.Tensor] = {}
        self._committed_source: str | None = None
        self._committed_epoch: int | None = None
        self._committed_forward_id = 0
        self._forward_id = 0
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current training epoch for deterministic residual schedules."""
        self._epoch = int(epoch)
        self.invalidate_diagnostics()

    def effective_residual_scale(self, epoch: int | None = None) -> float:
        """Return the fixed residual scale at a zero-based epoch index."""
        if self.residual_schedule == "constant":
            return self.residual_scale
        current = self._epoch if epoch is None else int(epoch)
        if current <= self.ramp_start_epoch:
            return 0.0
        if current >= self.ramp_end_epoch:
            return self.residual_scale
        progress = (current - self.ramp_start_epoch) / (self.ramp_end_epoch - self.ramp_start_epoch)
        return self.residual_scale * progress

    def estimated_gflops(self, image_size: int = 512) -> float:
        """Static multiply-add estimate for the cue branch at a square input."""
        h = image_size
        total = 2 * 1 * self.orientation_count * 3 * 3 * image_size * image_size
        for cin, cout, k, stride in ((self.orientation_count, self.encoder[0].conv.out_channels, 3, 2), (self.encoder[0].conv.out_channels, self.encoder[1].conv.out_channels, 3, 2), (self.encoder[1].conv.out_channels, self.encoder[2].conv.out_channels, 3, 1)):
            h = (h + 2 * (k // 2) - k) // stride + 1
            total += 2 * cin * cout * k * k * h * h
        total += 2 * self.projection.in_channels * self.projection.out_channels * h * h
        if self.gate is not None:
            total += 2 * self.gate.in_features * self.gate.out_features
        return total / 1e9

    def forward(self, p2: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        responses = oriented_edge_responses(image).to(device=p2.device, dtype=p2.dtype)
        # Gate from pooled directional responses; softmax guarantees normalized weights.
        pooled = responses.abs().mean(dim=(-2, -1))
        if self.orientation_gate_mode == "uniform":
            weights = torch.full(
                (responses.shape[0], self.orientation_count),
                1.0 / self.orientation_count,
                device=responses.device,
                dtype=responses.dtype,
            )
        else:
            weights = torch.softmax(self.gate(pooled), dim=-1)
        selected = responses * weights.unsqueeze(-1).unsqueeze(-1)
        feature = self.encoder(selected)
        if feature.shape[-2:] != p2.shape[-2:]:
            feature = F.interpolate(feature, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        effective_scale = self.effective_residual_scale()
        residual = self.projection(feature) * effective_scale
        rgb_norm = p2.detach().flatten(1).norm(dim=1).mean().clamp_min(1e-8)
        edge_norm = residual.detach().flatten(1).norm(dim=1).mean()
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        stats = {
            "orientation_gate_weights": weights.detach().mean(dim=0),
            "orientation_entropy": entropy.detach(),
            "edge_feature_norm": edge_norm.detach(),
            "rgb_p2_norm": rgb_norm,
            "edge_rgb_norm_ratio": edge_norm / rgb_norm,
            "edge_projection_norm": self.projection.weight.detach().norm(),
            "edge_activation_mean": feature.detach().mean(),
            "edge_activation_std": feature.detach().std(unbiased=False),
            "edge_residual_alpha": torch.as_tensor(effective_scale, device=p2.device, dtype=p2.dtype),
        }
        self._forward_id += 1
        source = "train" if self.training and torch.is_grad_enabled() else "eval"
        # Keep the old public field for callers that inspect it, but only the
        # phase-specific committed snapshot is used by diagnostics.
        self.last_stats = stats
        if source == "train":
            self._committed_stats = stats
            self._committed_source = source
            self._committed_epoch = self._epoch
            self._committed_forward_id = self._forward_id
        return p2 + residual

    def invalidate_diagnostics(self) -> None:
        """Invalidate a pending training snapshot at a controlled boundary."""
        self._committed_stats = {}
        self._committed_source = None
        self._committed_epoch = None
        self._committed_forward_id = 0

    def diagnostic_provenance(self) -> dict[str, int | str | None]:
        """Return provenance for the committed snapshot without tensor state."""
        return {
            "source": self._committed_source,
            "epoch": self._committed_epoch,
            "forward_id": self._committed_forward_id,
        }

    def diagnostic_metrics(self, source: str = "train") -> dict[str, float]:
        """Return scalar metrics from an explicitly selected committed phase.

        ``source='train'`` is the default used by the loss logger.  Evaluation
        and dummy/profile forwards never replace this snapshot, so a late
        forward cannot create a synthetic epoch row.
        """
        if source != "train" or self._committed_source != source or not self._committed_stats:
            return {}
        metrics: dict[str, float] = {}
        for name, value in self._committed_stats.items():
            if name == "orientation_gate_weights":
                for index, weight in enumerate(value.detach().flatten().tolist()):
                    metrics[f"edge_orientation_gate_weight_{index}"] = float(weight)
                continue
            metric_name = {
                "orientation_entropy": "edge_orientation_entropy",
                "edge_feature_norm": "edge_residual_norm",
                "rgb_p2_norm": "edge_p2_norm",
                "edge_rgb_norm_ratio": "edge_residual_p2_norm_ratio",
                "edge_projection_norm": "edge_projection_weight_norm",
                "edge_activation_mean": "edge_feature_activation_mean",
                "edge_activation_std": "edge_feature_activation_std",
                "edge_residual_alpha": "edge_effective_residual_alpha",
            }.get(name, f"edge_{name}")
            metrics[metric_name] = float(value.detach().mean().item())
        return metrics
