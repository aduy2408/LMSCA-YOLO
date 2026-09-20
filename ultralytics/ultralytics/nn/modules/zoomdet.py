"""ZoomDet integration boundary and provenance-safe geometry helpers.

The official ``twangnh/zoomdet_yolo`` implementation is tightly coupled to
MMYOLO's detector/grid/interpolator stack. This repository currently does not
vendor that stack. We therefore expose only validated geometry scaffolding and
explicitly block AZ training until the complete upstream offset predictor, grid,
box objective, and inverse prediction path are ported together. This file must
never silently fall back to the historical ``AdaptiveZoom`` crop-resize path.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


STATUS = "BLOCKED_EXACT_PORT"
UPSTREAM_REPOSITORY = "https://github.com/twangnh/zoomdet_yolo"


@dataclass(frozen=True)
class ZoomDetPortStatus:
    status: str = STATUS
    reason: str = (
        "Exact MMYOLO offset/grid/box-objective/inverse path is not vendored; "
        "the legacy AdaptiveZoom crop-resize transform is explicitly forbidden."
    )
    upstream_repository: str = UPSTREAM_REPOSITORY


def identity_box_transform(boxes: torch.Tensor) -> torch.Tensor:
    """Scaffold transform used by geometry tests; no image zoom is performed."""
    return boxes.clone()


def inverse_box_transform(boxes: torch.Tensor) -> torch.Tensor:
    """Scaffold inverse matching :func:`identity_box_transform`."""
    return boxes.clone()


def assert_exact_port_available() -> None:
    raise RuntimeError(
        "AZ_H1/AZ_H2 are BLOCKED_EXACT_PORT: do not substitute AdaptiveZoom or an invented warp."
    )
