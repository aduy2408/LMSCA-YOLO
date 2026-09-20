# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT, box_iou
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (
    RotatedTaskAlignedAssigner,
    SupportAwareTaskAlignedAssigner,
    TaskAlignedAssigner,
    dist2bbox,
    dist2rbox,
    make_anchors,
)
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


def _level_label(stride: float) -> str:
    """Return a stable P-level label from an actual Detect stride."""
    stride = float(stride)
    exponent = round(math.log2(stride)) if stride > 0 else -1
    return f"p{exponent}" if 2**exponent == stride else f"stride{int(stride)}"


def localization_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    lambda_ld: float,
) -> torch.Tensor:
    """Detached teacher-to-student DFL distribution KL used by N3."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Localization Distillation requires equal student and teacher DFL shapes")
    if lambda_ld < 0:
        raise ValueError("lambda_ld must be non-negative")
    if lambda_ld == 0 or not positive_mask.any():
        return student_logits.sum() * 0.0
    teacher_prob = F.softmax(teacher_logits.float(), dim=-1).detach()
    divergence = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1), teacher_prob, reduction="none"
    ).sum(-1)
    return divergence[positive_mask].mean().to(student_logits.dtype) * float(lambda_ld)


@dataclass(frozen=True)
class BoundaryContrastiveLossConfig:
    """YOLO.train kwargs for the boundary-aware contrastive localization loss."""

    gain: float = 0.05
    levels: int = 2
    ring: float = 1.0
    samples: int = 16
    tau: float = 0.2
    shrinkage: float = 0.25

    def as_train_kwargs(self) -> dict[str, float | int]:
        """Return kwargs accepted by YOLO.train(...)."""

        return {
            "boundary_contrast": float(self.gain),
            "boundary_levels": int(self.levels),
            "boundary_ring": float(self.ring),
            "boundary_samples": int(self.samples),
            "boundary_tau": float(self.tau),
            "boundary_shrinkage": float(self.shrinkage),
        }


def boundary_contrastive_loss_kwargs(
    gain: float = 0.05,
    levels: int = 2,
    ring: float = 1.0,
    samples: int = 16,
    tau: float = 0.2,
    shrinkage: float = 0.25,
) -> dict[str, float | int]:
    """Build YOLO.train kwargs that enable the boundary contrastive loss."""

    return BoundaryContrastiveLossConfig(
        gain=gain,
        levels=levels,
        ring=ring,
        samples=samples,
        tau=tau,
        shrinkage=shrinkage,
    ).as_train_kwargs()


def add_boundary_contrastive_loss(
    train_kwargs: dict | None = None,
    *,
    gain: float = 0.05,
    levels: int = 2,
    ring: float = 1.0,
    samples: int = 16,
    tau: float = 0.2,
    shrinkage: float = 0.25,
) -> dict:
    """Return train kwargs with boundary contrastive localization loss enabled."""

    kwargs = dict(train_kwargs or {})
    kwargs.update(
        boundary_contrastive_loss_kwargs(
            gain=gain,
            levels=levels,
            ring=ring,
            samples=samples,
            tau=tau,
            shrinkage=shrinkage,
        )
    )
    return kwargs


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, pos_q_weight: bool = True):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_q_weight = pos_q_weight

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor, cls_weights: torch.Tensor = None) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        pos_weight = gt_score if self.pos_q_weight else label
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + pos_weight * label
        if cls_weights is not None:
            weight = weight * cls_weights
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class GHMCClassificationLoss(nn.Module):
    """GHM-C style BCE for a dense detector (Li et al., AAAI 2019).

    The harmonizing weight is computed from the detached gradient magnitude
    ``|sigmoid(logit) - target|`` in a minibatch. This is deliberately a full
    classification-loss alternative, including negatives; it is not combined
    with FTSC positive evidence weights.
    """

    def __init__(self, bins: int = 10, momentum: float = 0.75) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError("GHM-C requires at least two gradient bins.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("GHM-C momentum must lie in [0, 1).")
        self.bins = int(bins)
        self.momentum = float(momentum)
        self.register_buffer("acc_sum", torch.zeros(self.bins), persistent=True)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        normalizer: torch.Tensor | float,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return GHM-weighted BCE on YOLO's native classification-loss scale.

        GHM determines the *relative* contribution of dense anchors, whereas
        ``normalizer`` preserves YOLO's ``sum(target_scores)`` normalization.
        Without this final normalization, the vast number of background
        anchors changes the classification-to-box/DFL loss ratio and makes an
        R3 comparison with the standard YOLO objective invalid.
        """
        with torch.no_grad():
            grad = (logits.sigmoid() - targets).abs()
            bin_index = (grad * self.bins).long().clamp_max(self.bins - 1)
            counts = torch.bincount(bin_index.reshape(-1), minlength=self.bins).to(dtype=logits.dtype)
            self.acc_sum.mul_(self.momentum).add_(counts.to(self.acc_sum), alpha=1.0 - self.momentum)
            effective_counts = torch.where(self.acc_sum > 0, self.acc_sum, counts.to(self.acc_sum)).clamp_min(1.0)
            weights = logits.numel() / effective_counts[bin_index]
            weights = weights / weights.mean().clamp_min(1e-9)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights
        if class_weights is not None:
            loss *= class_weights
        return loss.sum() / normalizer


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(
        self, pred_dist: torch.Tensor, target: torch.Tensor, bin_values: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        if bin_values is not None:
            values = bin_values.to(device=target.device, dtype=target.dtype)
            target = target.clamp(values[0], values[-1] - 0.01)
            right = torch.searchsorted(values, target, right=False).clamp(1, len(values) - 1)
            left = right - 1
            left_value, right_value = values[left], values[right]
            weight_right = (target - left_value) / (right_value - left_value)
            weight_left = 1 - weight_right
            losses = (
                F.cross_entropy(pred_dist, left.view(-1), reduction="none").view(left.shape) * weight_left
                + F.cross_entropy(pred_dist, right.view(-1), reduction="none").view(right.shape) * weight_right
            )
            return losses.mean(-1, keepdim=True)
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class WiseIouLoss(nn.Module):
    """Wise-IoU family losses for xyxy boxes with optional dynamic focusing."""

    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype: str = "WIoU", monotonous: bool | None = False, eps: float = 1e-7):
        """Initialize Wise-IoU loss.

        Args:
            ltype: IoU loss type, e.g. ``WIoU``, ``IoU``, ``GIoU``, ``DIoU``, ``CIoU``, ``EIoU`` or ``SIoU``.
            monotonous: ``True`` for monotonic focusing, ``False`` for non-monotonic focusing, ``None`` to disable it.
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.ltype = ltype.upper()
        if self.ltype == "WIOU":
            self.ltype = "WIoU"
        elif self.ltype.endswith("IOU"):
            self.ltype = self.ltype[:-3] + "IoU"
        if getattr(self, f"_{self.ltype}", None) is None:
            raise ValueError(f"Unsupported bbox_iou_loss='{ltype}'.")
        self.monotonous = monotonous
        self.eps = eps
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def __getitem__(self, item: str) -> torch.Tensor:
        """Evaluate cached geometry terms lazily."""
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-box IoU-family loss and plain IoU for xyxy boxes."""
        self._fget = {
            "pred": pred,
            "target": target,
            "pred_xy": lambda: (self["pred"][..., :2] + self["pred"][..., 2:4]) / 2,
            "pred_wh": lambda: (self["pred"][..., 2:4] - self["pred"][..., :2]).clamp(min=self.eps),
            "target_xy": lambda: (self["target"][..., :2] + self["target"][..., 2:4]) / 2,
            "target_wh": lambda: (self["target"][..., 2:4] - self["target"][..., :2]).clamp(min=self.eps),
            "min_coord": lambda: torch.minimum(self["pred"][..., :4], self["target"][..., :4]),
            "max_coord": lambda: torch.maximum(self["pred"][..., :4], self["target"][..., :4]),
            "wh_inter": lambda: torch.relu(self["min_coord"][..., 2:4] - self["max_coord"][..., :2]),
            "s_inter": lambda: torch.prod(self["wh_inter"], dim=-1),
            "s_union": lambda: torch.prod(self["pred_wh"], dim=-1)
            + torch.prod(self["target_wh"], dim=-1)
            - self["s_inter"]
            + self.eps,
            "wh_box": lambda: (self["max_coord"][..., 2:4] - self["min_coord"][..., :2]).clamp(min=self.eps),
            "s_box": lambda: torch.prod(self["wh_box"], dim=-1) + self.eps,
            "l2_box": lambda: torch.square(self["wh_box"]).sum(dim=-1) + self.eps,
            "d_center": lambda: self["pred_xy"] - self["target_xy"],
            "l2_center": lambda: torch.square(self["d_center"]).sum(dim=-1),
            "iou": lambda: self["s_inter"] / self["s_union"],
            "iou_loss": lambda: 1.0 - self["iou"],
        }

        if self.training:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self["iou_loss"].detach().mean())

        loss = self._scaled_loss(getattr(self, f"_{self.ltype}")())
        iou = self["iou"]
        delattr(self, "_fget")
        return loss, iou

    def _scaled_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply Wise-IoU dynamic focusing when configured."""
        if isinstance(self.monotonous, bool):
            beta = self["iou_loss"].detach() / self.iou_mean.clamp(min=self.eps)
            if self.monotonous:
                loss = loss * beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss = loss * beta / divisor
        return loss

    def _IoU(self) -> torch.Tensor:
        return self["iou_loss"]

    def _WIoU(self) -> torch.Tensor:
        dist = torch.exp(self["l2_center"] / self["l2_box"].detach())
        return dist * self["iou_loss"]

    def _EIoU(self) -> torch.Tensor:
        penalty = self["l2_center"] / self["l2_box"] + torch.square(self["d_center"] / self["wh_box"]).sum(dim=-1)
        return self["iou_loss"] + penalty

    def _GIoU(self) -> torch.Tensor:
        return self["iou_loss"] + (self["s_box"] - self["s_union"]) / self["s_box"]

    def _DIoU(self) -> torch.Tensor:
        return self["iou_loss"] + self["l2_center"] / self["l2_box"]

    def _CIoU(self) -> torch.Tensor:
        v = (
            4
            / math.pi**2
            * (
                torch.atan(self["pred_wh"][..., 0] / self["pred_wh"][..., 1])
                - torch.atan(self["target_wh"][..., 0] / self["target_wh"][..., 1])
            ).pow(2)
        )
        alpha = v / (self["iou_loss"] + v + self.eps)
        return self["iou_loss"] + self["l2_center"] / self["l2_box"] + alpha.detach() * v

    def _SIoU(self, theta: int = 4) -> torch.Tensor:
        angle = torch.arcsin(torch.abs(self["d_center"]).min(dim=-1)[0] / (self["l2_center"].sqrt() + self.eps))
        angle = torch.sin(2 * angle) - 2
        dist = angle[..., None] * torch.square(self["d_center"] / self["wh_box"])
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        d_shape = torch.abs(self["pred_wh"] - self["target_wh"])
        big_shape = torch.maximum(self["pred_wh"], self["target_wh"])
        w_shape = 1 - torch.exp(-d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(-d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape**theta + h_shape**theta
        return self["iou_loss"] + (dist + shape) / 2


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(
        self,
        reg_max: int = 16,
        iou_loss: str = "ciou",
        wiou_monotonous: bool | None = False,
        dfl_bins: torch.Tensor | None = None,
        pc_dfl_gain: float = 0.0,
        pc_dfl_margin: float = 1.5,
    ):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.register_buffer("dfl_bins", dfl_bins.clone().float() if dfl_bins is not None else None)
        self.pc_dfl_gain = float(pc_dfl_gain)
        self.pc_dfl_margin = float(pc_dfl_margin)
        self.last_p2_conflict: torch.Tensor | None = None
        self.last_pc_loss: torch.Tensor | None = None
        self.iou_loss_type = iou_loss.lower()
        if self.iou_loss_type not in {"ciou", "wiou", "iou", "giou", "diou", "eiou", "siou"}:
            raise ValueError(
                f"Unsupported bbox_iou_loss='{iou_loss}'. Expected one of: ciou, wiou, iou, giou, diou, eiou, siou."
            )
        self.wise_iou = (
            WiseIouLoss(ltype=iou_loss, monotonous=wiou_monotonous)
            if self.iou_loss_type != "ciou"
            else None
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
        box_weights: torch.Tensor | None = None,
        dfl_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if quality_weights is not None:
            weight = weight * quality_weights.to(device=weight.device, dtype=weight.dtype).view(-1, 1)
        iou_weight = weight
        if box_weights is not None:
            iou_weight = iou_weight * box_weights.to(device=weight.device, dtype=weight.dtype).view(-1, 1)
        if self.wise_iou is None:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * iou_weight).sum() / target_scores_sum
        else:
            iou_loss, _ = self.wise_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = (iou_loss.unsqueeze(-1) * iou_weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            positive_pred = pred_dist[fg_mask].view(-1, 4, self.dfl_loss.reg_max)
            positive_target = target_ltrb[fg_mask]
            self.last_p2_conflict = torch.zeros_like(fg_mask, dtype=positive_target.dtype)
            if self.dfl_bins is None:
                loss_dfl = self.dfl_loss(positive_pred.flatten(0, 1), positive_target)
            else:
                positive_is_p2 = (stride.view(-1) == stride.min()).unsqueeze(0).expand_as(fg_mask)[fg_mask]
                loss_dfl = positive_target.new_zeros((len(positive_target), 1))
                if positive_is_p2.any():
                    p2_pred, p2_target = positive_pred[positive_is_p2], positive_target[positive_is_p2]
                    loss_dfl[positive_is_p2] = self.dfl_loss(
                        p2_pred.flatten(0, 1),
                        p2_target,
                        self.dfl_bins,
                    )
                    if self.pc_dfl_gain > 0:
                        pc_loss, conflict = self.pair_competitive_loss(p2_pred, p2_target)
                        self.last_pc_loss = pc_loss.detach().mean()
                        loss_dfl[positive_is_p2] += self.pc_dfl_gain * pc_loss
                        dense_conflict = positive_target.new_zeros(len(positive_target))
                        dense_conflict[positive_is_p2] = conflict.detach()
                        self.last_p2_conflict[fg_mask] = dense_conflict
                if (~positive_is_p2).any():
                    loss_dfl[~positive_is_p2] = self.dfl_loss(
                        positive_pred[~positive_is_p2].flatten(0, 1), positive_target[~positive_is_p2]
                    )
            dfl_weight = weight
            if dfl_weights is not None:
                dfl_weight = dfl_weight * dfl_weights.to(device=weight.device, dtype=weight.dtype).view(-1, 1)
            loss_dfl = loss_dfl * dfl_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            dfl_weight = weight
            if dfl_weights is not None:
                dfl_weight = dfl_weight * dfl_weights.to(device=weight.device, dtype=weight.dtype).view(-1, 1)
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
                * dfl_weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    def pair_competitive_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make the interpolated target pair beat its two adjacent competitors."""
        values = self.dfl_bins.to(device=target.device, dtype=target.dtype)
        target = target.clamp(values[0], values[-1] - 0.01)
        right = torch.searchsorted(values, target, right=False).clamp(1, len(values) - 1)
        left = right - 1
        weight_right = (target - values[left]) / (values[right] - values[left])
        weight_left = 1.0 - weight_right
        target_score = torch.logsumexp(
            torch.stack(
                (
                    logits.gather(-1, left.unsqueeze(-1)).squeeze(-1) + weight_left.clamp_min(1e-9).log(),
                    logits.gather(-1, right.unsqueeze(-1)).squeeze(-1) + weight_right.clamp_min(1e-9).log(),
                )
            ),
            dim=0,
        )
        competitor_indices = torch.stack((left - 1, right + 1), dim=-1)
        valid = (competitor_indices >= 0) & (competitor_indices < len(values))
        competitor_score = logits.gather(-1, competitor_indices.clamp(0, len(values) - 1))
        competitor_score = torch.logsumexp(competitor_score.masked_fill(~valid, -torch.inf), dim=-1)
        per_side = F.softplus(self.pc_dfl_margin + competitor_score - target_score)
        return per_side.mean(-1, keepdim=True), per_side.max(-1).values


# EXPERIMENTAL: Boundary-aware contrastive loss for tiny-object localization.
# This auxiliary objective samples Detect feature maps around each GT box so
# object-interior features stay close while boundary/background features are
# pushed away. It is opt-in through boundary_contrast and has no inference path.
class BoundaryContrastiveLoss(nn.Module):
    """Boundary-aware InfoNCE loss over Detect feature maps."""

    def __init__(self, levels: int = 2, ring: float = 1.0, samples: int = 16, tau: float = 0.2, shrinkage: float = 0.25):
        """Initialize boundary contrast settings."""
        super().__init__()
        self.levels = max(int(levels), 0)
        self.ring = max(float(ring), 0.0)
        self.samples = max(int(samples), 1)
        self.tau = max(float(tau), 1e-6)
        self.shrinkage = max(float(shrinkage), 0.0)

    @staticmethod
    def _sample_indices(mask: torch.Tensor, limit: int) -> torch.Tensor:
        """Sample up to limit flattened indices from a boolean mask."""
        idx = mask.flatten().nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > limit:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:limit]]
        return idx

    def forward(
        self,
        feats: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary-aware contrastive loss from feature maps and image-space GT boxes."""
        losses = []
        num_levels = min(self.levels, len(feats), len(strides))
        for level, feat in enumerate(feats[:num_levels]):
            bs, _, h, w = feat.shape
            stride = strides[level].to(device=feat.device, dtype=feat.dtype).clamp(min=1)
            y, x = torch.meshgrid(
                torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5,
                torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5,
                indexing="ij",
            )

            for bi in range(bs):
                boxes = gt_bboxes[bi][mask_gt[bi, :, 0].bool()]
                if boxes.numel() == 0:
                    continue

                fmap = feat[bi].flatten(1).transpose(0, 1)
                fmap = F.normalize(fmap, dim=1)

                for box in boxes:
                    x1, y1, x2, y2 = box / stride
                    x1 = x1.clamp(0, w)
                    x2 = x2.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    y2 = y2.clamp(0, h)
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    # EXPERIMENTAL: prefer a tighter interior region when it
                    # exists, so positives represent object texture, not edges.
                    pad_x = torch.minimum((x2 - x1) * self.shrinkage, torch.tensor(0.5, device=feat.device, dtype=feat.dtype))
                    pad_y = torch.minimum((y2 - y1) * self.shrinkage, torch.tensor(0.5, device=feat.device, dtype=feat.dtype))
                    inner = (x >= x1 + pad_x) & (x < x2 - pad_x) & (y >= y1 + pad_y) & (y < y2 - pad_y)
                    obj = inner if inner.any() else ((x >= x1) & (x < x2) & (y >= y1) & (y < y2))

                    dx1 = (x1 - self.ring).clamp(0, w)
                    dy1 = (y1 - self.ring).clamp(0, h)
                    dx2 = (x2 + self.ring).clamp(0, w)
                    dy2 = (y2 + self.ring).clamp(0, h)
                    dilated = (x >= dx1) & (x < dx2) & (y >= dy1) & (y < dy2)
                    original = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                    boundary = dilated & ~original

                    # EXPERIMENTAL: nearby background is a cheap hard-negative
                    # proxy without using prediction confidence.
                    near_x1 = (x1 - self.ring * 3).clamp(0, w)
                    near_y1 = (y1 - self.ring * 3).clamp(0, h)
                    near_x2 = (x2 + self.ring * 3).clamp(0, w)
                    near_y2 = (y2 + self.ring * 3).clamp(0, h)
                    near = (x >= near_x1) & (x < near_x2) & (y >= near_y1) & (y < near_y2)
                    background = near & ~dilated
                    if not background.any():
                        background = ~dilated

                    obj_idx = self._sample_indices(obj, self.samples)
                    bnd_idx = self._sample_indices(boundary, self.samples)
                    bg_idx = self._sample_indices(background, self.samples)
                    if obj_idx.numel() < 2 or bnd_idx.numel() == 0 or bg_idx.numel() == 0:
                        continue

                    obj_feat = fmap[obj_idx]
                    pos = obj_feat.roll(1, dims=0)
                    neg = fmap[torch.cat((bnd_idx, bg_idx), 0)]

                    pos_logits = (obj_feat * pos).sum(1, keepdim=True) / self.tau
                    neg_logits = obj_feat @ neg.T / self.tau
                    logits = torch.cat((pos_logits, neg_logits), 1)
                    labels = torch.zeros(logits.shape[0], device=feat.device, dtype=torch.long)
                    losses.append(F.cross_entropy(logits, labels))

        return torch.stack(losses).mean() if losses else gt_bboxes.sum() * 0.0


# EXPERIMENTAL: Localization Quality Map loss for tiny-object localization.
# This auxiliary objective teaches train-only 1x1 heads to predict a smooth
# center-high localization target for each GT box. It has no inference path.
class LocalizationQualityLoss(nn.Module):
    """Gaussian localization quality map supervision over Detect feature maps."""

    def __init__(self, levels: int = 2, sigma: float = 0.45, loss_type: str = "mse"):
        """Initialize LQM settings."""
        super().__init__()
        self.levels = max(int(levels), 0)
        self.sigma = max(float(sigma), 1e-3)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"mse", "smoothl1"}:
            raise ValueError("loc_quality_loss must be 'mse' or 'smoothl1'.")

    def forward(
        self,
        loc_maps: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LQM loss from predicted maps and image-space GT boxes."""
        losses = []
        num_levels = min(self.levels, len(loc_maps), len(strides))
        for level, loc_map in enumerate(loc_maps[:num_levels]):
            bs, _, h, w = loc_map.shape
            stride = strides[level].to(device=loc_map.device, dtype=loc_map.dtype).clamp(min=1)
            y, x = torch.meshgrid(
                torch.arange(h, device=loc_map.device, dtype=loc_map.dtype) + 0.5,
                torch.arange(w, device=loc_map.device, dtype=loc_map.dtype) + 0.5,
                indexing="ij",
            )
            target = torch.zeros((bs, 1, h, w), device=loc_map.device, dtype=loc_map.dtype)

            for bi in range(bs):
                boxes = gt_bboxes[bi][mask_gt[bi, :, 0].bool()]
                if boxes.numel() == 0:
                    continue
                image_target = target[bi, 0]
                for box in boxes:
                    x1, y1, x2, y2 = box / stride
                    x1 = x1.clamp(0, w)
                    x2 = x2.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    y2 = y2.clamp(0, h)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 1 or bh < 1:
                        continue

                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    sigma_x = (bw * self.sigma).clamp(min=1e-3)
                    sigma_y = (bh * self.sigma).clamp(min=1e-3)
                    inside = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                    quality = torch.exp(
                        -0.5 * (((x - cx) / sigma_x).pow(2) + ((y - cy) / sigma_y).pow(2))
                    ) * inside.to(loc_map.dtype)
                    image_target.copy_(torch.maximum(image_target, quality))

            pred = loc_map.sigmoid()
            if self.loss_type == "smoothl1":
                losses.append(F.smooth_l1_loss(pred, target))
            else:
                losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean() if losses else gt_bboxes.sum() * 0.0


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        enable_ftsc: bool = True,
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.model = model
        self.assignment_override: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.last_assignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        m = model.model[-1]  # Detect() module
        self.capture_assignment = bool(getattr(m, "hbs_enabled", False))
        self.ftsc_calibrator = getattr(m, "ftsc_calibrator", None) if enable_ftsc else None
        self.ftsc_metrics: dict[str, float] = {}
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.vfl = (
            VarifocalLoss(
                gamma=float(getattr(h, "vfl_gamma", 2.0)),
                alpha=float(getattr(h, "vfl_alpha", 0.75)),
                pos_q_weight=bool(getattr(h, "vfl_pos_q_weight", True)),
            )
            if bool(getattr(h, "vfl", False))
            else None
        )
        ghm = dict(getattr(m, "ghm_config", {}))
        self.ghm_cls = bool(ghm.get("enabled", False))
        self.ghm_cls_loss = (
            GHMCClassificationLoss(
                bins=int(ghm.get("bins", 10)),
                momentum=float(ghm.get("momentum", 0.75)),
            ).to(device)
            if self.ghm_cls
            else None
        )
        if self.ghm_cls and self.vfl is not None:
            raise ValueError("GHM-C and Varifocal Loss are mutually exclusive classification objectives.")
        if self.ghm_cls and bool(getattr(h, "cls_iou_target", False)):
            raise ValueError("GHM-C uses standard TAL targets; disable cls_iou_target.")
        if self.ghm_cls and self.ftsc_calibrator is not None:
            raise ValueError("GHM-C replaces FTSC positive reweighting; configure ftsc.enabled=false.")
        if self.ftsc_calibrator is not None and self.ftsc_calibrator.classification_replacement == "deim_mal":
            if self.vfl is not None or bool(getattr(h, "cls_iou_target", False)):
                raise ValueError("DEIM-MAL requires the default TAL-BCE classification path.")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.localization_distill_enabled = bool(getattr(h, "localization_distill", False))
        self.localization_distill_lambda = float(getattr(h, "localization_distill_lambda", 0.1))
        self.localization_distill_teacher = getattr(model, "_ftsc_localization_teacher", None)
        if self.localization_distill_enabled and self.localization_distill_teacher is None:
            raise ValueError("localization_distill requires an explicit frozen teacher attached by the runner")
        if self.localization_distill_enabled and self.localization_distill_lambda < 0:
            raise ValueError("localization_distill_lambda must be non-negative")
        if self.localization_distill_teacher is not None:
            self.localization_distill_teacher.to(device)
            self.localization_distill_teacher.eval()
            for parameter in self.localization_distill_teacher.parameters():
                parameter.requires_grad_(False)
        self.localization_distill_metrics: dict[str, float] = {}

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        assigner_kwargs = {
            "topk": int(getattr(h, "tal_topk", tal_topk)),
            "num_classes": self.nc,
            "alpha": float(getattr(h, "tal_alpha", 0.5)),
            "beta": float(getattr(h, "tal_beta", 6.0)),
            "stride": self.stride.tolist(),
            "topk2": tal_topk2,
        }
        self.assigner = (
            SupportAwareTaskAlignedAssigner(
                **assigner_kwargs,
                support_topk=int(getattr(h, "support_assignment_topk", 5)),
            )
            if bool(getattr(h, "support_assignment", False))
            else TaskAlignedAssigner(**assigner_kwargs)
        )
        self.support_assignment_metrics: dict[str, float] = {}
        self.bbox_loss = BboxLoss(
            m.reg_max,
            iou_loss=getattr(h, "bbox_iou_loss", "ciou"),
            wiou_monotonous=getattr(h, "wiou_monotonous", False),
            dfl_bins=getattr(m, "p2_dfl_bins", getattr(m, "p3_dfl_bins", None)),
            pc_dfl_gain=float(getattr(h, "pc_dfl_gain", 0.0)) if hasattr(m, "p2_dfl_bins") else 0.0,
            pc_dfl_margin=float(getattr(h, "pc_dfl_margin", 1.5)),
        ).to(device)
        # EXPERIMENTAL: optional boundary-aware contrastive objective for
        # Varroa-scale localization. Disabled by default via boundary_contrast=0.
        self.boundary_gain = float(getattr(h, "boundary_contrast", 0.0))
        self.boundary_loss = (
            BoundaryContrastiveLoss(
                levels=getattr(h, "boundary_levels", 2),
                ring=getattr(h, "boundary_ring", 1.0),
                samples=getattr(h, "boundary_samples", 16),
                tau=getattr(h, "boundary_tau", 0.2),
                shrinkage=getattr(h, "boundary_shrinkage", 0.25),
            ).to(device)
            if self.boundary_gain > 0
            else None
        )
        # EXPERIMENTAL: optional train-only Localization Quality Map supervision.
        # The Detect module owns the 1x1 heads so optimizer updates them.
        self.loc_quality_gain = float(getattr(h, "loc_quality", 0.0))
        m.loc_quality_enabled = self.loc_quality_gain > 0
        self.loc_quality_heads = getattr(m, "loc_cv", None)
        self.loc_quality_loss = (
            LocalizationQualityLoss(
                levels=getattr(h, "loc_quality_levels", 2),
                sigma=getattr(h, "loc_quality_sigma", 0.45),
                loss_type=getattr(h, "loc_quality_loss", "mse"),
            ).to(device)
            if self.loc_quality_gain > 0
            else None
        )
        self.quality_head = bool(getattr(m, "quality_head", getattr(h, "quality_head", False)))
        self.quality_loss_type = str(getattr(h, "quality_loss", "bce")).lower()
        if self.quality_loss_type not in {"bce", "bce_balanced", "l1"}:
            raise ValueError("quality_loss must be 'bce', 'bce_balanced', or 'l1'.")
        self.quality_gain = float(getattr(h, "quality_gain", 0.5))
        self.quality_neg_gain = float(getattr(h, "quality_neg_gain", 0.10))
        self.quality_pos_iou_thr = float(getattr(h, "quality_pos_iou_thr", 0.5))
        self.quality_hard_neg_iou_thr = float(getattr(h, "quality_hard_neg_iou_thr", 0.3))
        self.quality_hard_neg_score_thr = float(getattr(h, "quality_hard_neg_score_thr", 0.05))
        self.quality_target_mode = str(getattr(h, "quality_target_mode", "iou_power")).lower()
        if self.quality_target_mode not in {"iou_power", "ap75_ramp"}:
            raise ValueError("quality_target_mode must be 'iou_power' or 'ap75_ramp'.")
        self.quality_target_power = float(getattr(h, "quality_target_power", 1.0))
        self.quality_ramp_low = float(getattr(h, "quality_ramp_low", 0.50))
        self.quality_ramp_high = float(getattr(h, "quality_ramp_high", 0.75))
        if self.quality_ramp_high <= self.quality_ramp_low:
            raise ValueError("quality_ramp_high must be greater than quality_ramp_low.")
        self.quality_neg_mode = str(getattr(h, "quality_neg_mode", "hard")).lower()
        if self.quality_neg_mode not in {"hard", "all"}:
            raise ValueError("quality_neg_mode must be 'hard' or 'all'.")
        self.quality_detach_target = bool(getattr(h, "quality_detach_target", True))
        self.quality_debug = bool(getattr(h, "quality_debug", False))
        self.quality_debug_batches = int(getattr(h, "quality_debug_batches", 1))
        self.quality_debug_seen = 0
        self.dgfe_rec_gain = float(getattr(h, "dgfe_rec_gain", 0.0))
        self.dgfe_spatial_gain = float(getattr(h, "dgfe_spatial_gain", 0.0))
        self.dgfe_spatial_warmup_epochs = max(int(getattr(h, "dgfe_spatial_warmup_epochs", 0)), 0)
        self.dgfe_spatial_warmup_start = max(
            min(float(getattr(h, "dgfe_spatial_warmup_start", 0.1)), 1.0), 0.0
        )
        self.dgfe_boundary_ring = float(getattr(h, "dgfe_boundary_ring", 1.0))
        self.dgfe_inner_value = float(getattr(h, "dgfe_inner_value", 0.3))
        self.dgfe_tiny_area = float(getattr(h, "dgfe_tiny_area", 4.0))
        self.dgfe_neg_pos_ratio = int(getattr(h, "dgfe_neg_pos_ratio", 3))
        self.dgfe_neg_gain = float(getattr(h, "dgfe_neg_gain", 0.25))
        self.dgfe_spatial_target_mode = str(getattr(h, "dgfe_spatial_target_mode", "iou")).lower()
        if self.dgfe_spatial_target_mode not in {"iou", "edge_error"}:
            raise ValueError("dgfe_spatial_target_mode must be 'iou' or 'edge_error'.")
        self.dgfe_edge_error_norm = max(float(getattr(h, "dgfe_edge_error_norm", 0.25)), 1e-9)
        self.p2_detail_rec_gain = float(getattr(h, "p2_detail_rec_gain", 0.0))
        self.p2_detail_metrics = {}
        self.rank_gain = float(getattr(h, "rank_loss", 0.0))
        self.rank_tau = float(getattr(h, "rank_tau", 0.25))
        self.rank_iou_margin = float(getattr(h, "rank_iou_margin", 0.10))
        self.rank_topk = int(getattr(h, "rank_topk", 10))
        self.positive_confidence_rescue_gain = float(getattr(h, "positive_confidence_rescue_gain", 0.0))
        self.positive_confidence_rescue_gamma = float(getattr(h, "positive_confidence_rescue_gamma", 1.0))
        if self.positive_confidence_rescue_gain < 0 or self.positive_confidence_rescue_gamma < 0:
            raise ValueError("positive confidence rescue gain and gamma must be non-negative")
        if self.positive_confidence_rescue_gain and (
            self.vfl is not None or bool(getattr(h, "cls_iou_target", False))
        ):
            raise ValueError("positive confidence rescue requires default TAL-BCE classification")
        self.positive_confidence_rescue_metrics = {}
        self.loc_assign = bool(getattr(h, "loc_assign", False))
        self.loc_assign_topk = int(getattr(h, "loc_assign_topk", 3))
        self.loc_assign_max_stride = float(getattr(h, "loc_assign_max_stride", 8.0))
        self.loc_assign_center_radius = float(getattr(h, "loc_assign_center_radius", 2.5))
        self.loc_assign_weight = float(getattr(h, "loc_assign_weight", 0.5))
        if self.ftsc_calibrator is not None and self.loc_assign:
            raise ValueError("Anchor-free FTSC is assignment-preserving and is incompatible with loc_assign.")
        if (
            self.ftsc_calibrator is not None
            and "dfl_distribution" in self.ftsc_calibrator.evidence_names
            and getattr(m, "p2_dfl_bins", getattr(m, "p3_dfl_bins", None)) is not None
        ):
            raise ValueError("FTSC DFL-distribution evidence currently requires uniform DFL bins.")
        self.detect_head = m
        self.box_consensus_gain = float(getattr(h, "box_consensus_gain", 0.0))
        self.box_consensus_warmup_start = int(getattr(h, "box_consensus_warmup_start", 5))
        self.box_consensus_warmup_end = int(getattr(h, "box_consensus_warmup_end", 15))
        self.box_consensus_log_grad_ratio = bool(getattr(h, "box_consensus_log_grad_ratio", False))
        self.consensus_metrics = {}
        self.psd_enabled = bool(getattr(h, "positive_support_dropout", False))
        self.psd_mode = str(getattr(h, "positive_support_mode", "dominant")).lower()
        self.psd_gain = float(getattr(h, "positive_support_gain", 0.0))
        self.psd_prob = float(getattr(h, "positive_support_prob", 0.5))
        self.psd_min_count = int(getattr(h, "positive_support_min_count", 3))
        self.psd_aux_topk = int(getattr(h, "positive_support_aux_topk", 3))
        self.psd_radius = int(getattr(h, "positive_support_radius", 2))
        self.psd_fill_kernel = int(getattr(h, "positive_support_fill_kernel", 3))
        self.psd_warmup_start = int(getattr(h, "positive_support_warmup_start", 5))
        self.psd_warmup_end = int(getattr(h, "positive_support_warmup_end", 15))
        self.psd_metrics = {}
        if self.psd_enabled:
            assert not getattr(m, "dfl_residual", False), "PSD is not compatible with dfl_residual"
            assert getattr(m, "p2_dfl_bins", None) is None, "PSD is not compatible with non-uniform DFL (NUDFL)"
            assert getattr(h, "bbox_iou_loss", "ciou") == "ciou", "PSD currently requires CIoU loss"
            assert not getattr(m, "box_detail_head", False), "PSD currently requires box_detail_head == False"
        self.epoch = 0
        self.dfl_residual = bool(getattr(m, "dfl_residual", False))
        self.dfl_residual_scale = float(getattr(m, "dfl_residual_scale", getattr(h, "dfl_residual_scale", 0.25)))
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    @staticmethod
    def _ftsc_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        """Return a finite detached Pearson correlation for FTSC diagnostics."""
        if left.numel() < 2 or right.numel() != left.numel():
            return 0.0
        left = left.detach().float().flatten()
        right = right.detach().float().flatten()
        left = left - left.mean()
        right = right - right.mean()
        denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
        if float(denominator.item()) <= 1e-12:
            return 0.0
        return float((left * right).sum().div(denominator).item())

    @staticmethod
    def _post_tal_support_metrics(
        mask_gt: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        flat_stride: torch.Tensor,
        strides: torch.Tensor,
    ) -> dict[str, float]:
        """Collect assignment support counts before any FTSC reweighting.

        ``mask_gt`` is built before TAL and therefore retains GTs with zero
        positives.  The returned values beginning with ``_tal_`` are raw
        sufficient statistics; DetectionModel aggregates these across batches
        and derives fractions only once at epoch end.
        """
        batch_size, max_gt = mask_gt.shape[:2]
        valid_gt = mask_gt.squeeze(-1).bool()
        gt_count = int(valid_gt.sum().item())
        positive_count = int(fg_mask.sum().item())
        if fg_mask.any():
            batch_ids = torch.arange(batch_size, device=fg_mask.device).view(-1, 1).expand_as(fg_mask)[fg_mask]
            gt_ids = target_gt_idx[fg_mask].long()
            linear = batch_ids * max_gt + gt_ids
            support_dense = torch.zeros(batch_size * max_gt, device=fg_mask.device, dtype=torch.long)
            support_dense.scatter_add_(0, linear, torch.ones_like(linear, dtype=torch.long))
            support_dense = support_dense.view(batch_size, max_gt)
        else:
            support_dense = torch.zeros((batch_size, max_gt), device=fg_mask.device, dtype=torch.long)
        support = support_dense[valid_gt]
        histogram = torch.bincount(support.clamp_max(16), minlength=17) if support.numel() else torch.zeros(17, device=fg_mask.device, dtype=torch.long)
        metrics: dict[str, float] = {
            "_tal_gt_count": float(gt_count),
            "_tal_positive_count": float(positive_count),
        }
        for index, value in enumerate(histogram.tolist()):
            metrics[f"_tal_support_bin_{index}"] = float(value)

        # Per-level support is derived from the same TAL foreground mask.  It
        # never changes assignment and remains valid for non-contiguous levels.
        level_counts: dict[int, torch.Tensor] = {}
        for stride in strides.detach().cpu().tolist():
            level_mask = flat_stride == float(stride)
            level_fg = fg_mask & level_mask.unsqueeze(0)
            if level_fg.any():
                level_batch = torch.arange(batch_size, device=fg_mask.device).view(-1, 1).expand_as(fg_mask)[level_fg]
                level_gt = target_gt_idx[level_fg].long()
                level_linear = level_batch * max_gt + level_gt
                dense = torch.zeros(batch_size * max_gt, device=fg_mask.device, dtype=torch.long)
                dense.scatter_add_(0, level_linear, torch.ones_like(level_linear, dtype=torch.long))
                level_counts[int(round(float(stride)))] = dense.view(batch_size, max_gt)
            else:
                level_counts[int(round(float(stride)))] = torch.zeros((batch_size, max_gt), device=fg_mask.device, dtype=torch.long)
            label = _level_label(float(stride))
            metrics[f"_tal_positive_count_{label}"] = float(level_counts[int(round(float(stride)))].sum().item())

        p2 = level_counts.get(4, torch.zeros_like(support_dense))[valid_gt]
        p3 = level_counts.get(8, torch.zeros_like(support_dense))[valid_gt]
        metrics["_tal_gt_support_p2_only_count"] = float(((p2 > 0) & (p3 == 0)).sum().item())
        metrics["_tal_gt_support_p3_only_count"] = float(((p2 == 0) & (p3 > 0)).sum().item())
        metrics["_tal_gt_support_p2_p3_count"] = float(((p2 > 0) & (p3 > 0)).sum().item())
        metrics["_tal_gt_support_ge2_count"] = float((support >= 2).sum().item())
        return metrics

    @staticmethod
    def _apply_ftsc_positive_cls_weights(
        cls_weights: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply positive FTSC weights only to target-class classification elements."""
        if positive_weights.numel() != int(fg_mask.sum().item()):
            raise ValueError("FTSC requires one classification weight per TAL positive.")
        dense_positive_weights = cls_weights.new_ones(fg_mask.shape)
        dense_positive_weights[fg_mask] = positive_weights.to(dtype=cls_weights.dtype)
        return cls_weights * torch.where(
            target_scores > 0,
            dense_positive_weights.unsqueeze(-1),
            torch.ones_like(cls_weights),
        )

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def positive_confidence_rescue_loss(
        self, pred_scores: torch.Tensor, target_scores: torch.Tensor, fg_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return low-target TAL-positive rescue loss and the selected targets/logits."""
        positive_targets, positive_classes = target_scores.max(dim=-1)
        positive_logits = pred_scores.gather(-1, positive_classes.unsqueeze(-1)).squeeze(-1)
        target = positive_targets[fg_mask].detach().to(pred_scores.dtype)
        logits = positive_logits[fg_mask]
        raw = ((1.0 - target).pow(self.positive_confidence_rescue_gamma) * F.softplus(-logits)).sum()
        return raw / fg_mask.sum().clamp_min(1), target, logits

    def bbox_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor,
        pred_residual: torch.Tensor | None = None,
        stride_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            probability = pred_dist.view(b, a, 4, c // 4).softmax(3)
            if self.bbox_loss.dfl_bins is not None:
                if stride_tensor is None:
                    raise ValueError("Non-uniform DFL decoding requires stride_tensor")
                uniform = self.proj.to(pred_dist.dtype).view(1, 1, 1, -1)
                custom = self.bbox_loss.dfl_bins.to(pred_dist.dtype).view(1, 1, 1, -1)
                is_finest = (stride_tensor.view(1, a, 1, 1) == stride_tensor.min()).to(pred_dist.dtype)
                values = uniform + is_finest * (custom - uniform)
                pred_dist = (probability * values).sum(3)
            else:
                pred_dist = probability.matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            if pred_residual is not None:
                pred_dist = (pred_dist + pred_residual.tanh() * self.dfl_residual_scale).clamp(
                    min=0, max=max(self.reg_max - 1, 0)
                )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def pairwise_ranking_loss(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rank TAL foreground candidates so higher-IoU boxes receive higher assigned-class logits."""
        if not fg_mask.any() or self.rank_topk < 2:
            return pred_scores.sum() * 0.0

        fg_indices = fg_mask.nonzero(as_tuple=False)
        batch_indices, anchor_indices = fg_indices[:, 0], fg_indices[:, 1]
        rank_pred_bboxes = pred_bboxes.detach()[fg_mask]
        rank_target_bboxes = target_bboxes.detach()[fg_mask]
        rank_iou = bbox_iou(rank_pred_bboxes, rank_target_bboxes, xywh=False, CIoU=False).squeeze(-1).clamp(0)

        class_indices = target_scores[fg_mask].argmax(-1)
        assigned_logits = pred_scores[batch_indices, anchor_indices, class_indices]
        group_ids = batch_indices * (target_gt_idx.max() + 1).clamp_min(1) + target_gt_idx[fg_mask]

        rank_losses = []
        rank_weights = []
        tau = max(self.rank_tau, 1e-6)
        for group_id in group_ids.unique():
            group_mask = group_ids == group_id
            if group_mask.sum() < 2:
                continue

            group_iou = rank_iou[group_mask]
            group_logits = assigned_logits[group_mask]
            topk = min(self.rank_topk, int(group_iou.numel()))
            order = group_iou.argsort(descending=True)[:topk]
            group_iou = group_iou[order]
            group_logits = group_logits[order]

            iou_gap = group_iou[:, None] - group_iou[None, :]
            pair_mask = iou_gap > self.rank_iou_margin
            if not pair_mask.any():
                continue

            score_gap = group_logits[:, None] - group_logits[None, :]
            weights = iou_gap[pair_mask].detach()
            rank_losses.append(F.softplus(-score_gap[pair_mask] / tau) * weights)
            rank_weights.append(weights)

        if not rank_losses:
            return pred_scores.sum() * 0.0
        rank_losses = torch.cat(rank_losses)
        rank_weights = torch.cat(rank_weights)
        return rank_losses.sum() / (rank_weights.sum() + 1e-9)

    def build_localization_targets(
        self,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand box/DFL positives with GT-center local anchors without changing cls targets."""
        if not self.loc_assign or self.loc_assign_topk <= 0:
            return target_bboxes, target_scores, fg_mask

        loc_bboxes = target_bboxes.clone()
        loc_scores = target_scores.clone()
        loc_mask = fg_mask.clone()
        anchor_pixel = anchor_points * stride_tensor
        stride_flat = stride_tensor.squeeze(-1)
        stride_ok = stride_flat <= self.loc_assign_max_stride if self.loc_assign_max_stride > 0 else torch.ones_like(
            stride_flat, dtype=torch.bool
        )
        if not stride_ok.any():
            return loc_bboxes, loc_scores, loc_mask

        nc = loc_scores.shape[-1]
        for bi in range(target_bboxes.shape[0]):
            valid = mask_gt[bi].squeeze(-1).bool()
            if not valid.any():
                continue
            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
            for gi in valid_indices:
                gt_box = gt_bboxes[bi, gi]
                gt_c = (gt_box[:2] + gt_box[2:]) * 0.5
                dist = (anchor_pixel - gt_c).pow(2).sum(-1).sqrt()
                radius = self.loc_assign_center_radius * stride_flat
                cand = stride_ok & (dist <= radius)
                if not cand.any():
                    continue
                cand_idx = cand.nonzero(as_tuple=False).squeeze(1)
                norm_dist = dist[cand_idx] / stride_flat[cand_idx].clamp_min(1e-6)
                topk = min(self.loc_assign_topk, int(norm_dist.numel()))
                if topk <= 0:
                    continue
                chosen = cand_idx[norm_dist.topk(topk, largest=False).indices]
                cls_idx = int(gt_labels[bi, gi, 0].clamp(0, nc - 1).item())
                new_only = ~loc_mask[bi, chosen]
                if not new_only.any():
                    continue
                chosen = chosen[new_only]
                loc_mask[bi, chosen] = True
                loc_bboxes[bi, chosen] = gt_box
                loc_scores[bi, chosen, cls_idx] = torch.maximum(
                    loc_scores[bi, chosen, cls_idx],
                    loc_scores.new_full((chosen.numel(),), self.loc_assign_weight),
                )
        return loc_bboxes, loc_scores, loc_mask

    def _box_consensus_current_gain(self) -> float:
        """Linearly ramp consensus after an initial zero-gain warmup."""
        if self.epoch <= self.box_consensus_warmup_start:
            return 0.0
        if self.box_consensus_warmup_end <= self.box_consensus_warmup_start:
            return self.box_consensus_gain
        progress = min(
            (self.epoch - self.box_consensus_warmup_start)
            / (self.box_consensus_warmup_end - self.box_consensus_warmup_start),
            1.0,
        )
        return self.box_consensus_gain * progress

    def _psd_current_gain(self) -> float:
        """Linearly ramp PSD after an initial zero-gain warmup."""
        if self.epoch <= self.psd_warmup_start:
            return 0.0
        if self.psd_warmup_end <= self.psd_warmup_start:
            return self.psd_gain
        progress = min(
            (self.epoch - self.psd_warmup_start)
            / (self.psd_warmup_end - self.psd_warmup_start),
            1.0,
        )
        return self.psd_gain * progress

    def box_consensus_loss(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        n_p2: int,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Penalize edge-error dispersion among original TAL-positive P2 anchors per GT."""
        group_losses, counts = [], []
        eligible = 0
        for batch_idx in range(fg_mask.shape[0]):
            valid_gt = int(mask_gt[batch_idx].sum().item())
            p2_fg = fg_mask[batch_idx, :n_p2]
            assigned = target_gt_idx[batch_idx, :n_p2]
            for gt_idx in range(valid_gt):
                group = p2_fg & (assigned == gt_idx)
                count = int(group.sum().item())
                counts.append(count)
                if count < 2:
                    continue
                eligible += 1
                pred = pred_bboxes[batch_idx, :n_p2][group]
                target = target_bboxes[batch_idx, :n_p2][group]
                wh = (target[0, 2:] - target[0, :2]).clamp_min(1e-6)
                scale = torch.cat((wh, wh))
                error = (pred - target) / scale
                centered = error - error.mean(0, keepdim=True)
                group_losses.append(F.smooth_l1_loss(centered, torch.zeros_like(centered), reduction="mean"))
        raw = torch.stack(group_losses).mean() if group_losses else pred_bboxes.sum() * 0.0
        counts_tensor = torch.tensor(counts, device=pred_bboxes.device, dtype=pred_bboxes.dtype)
        metrics = {
            "consensus_gt_coverage": eligible / max(len(counts), 1),
            "consensus_p2_pos_median": float(counts_tensor.median().item()) if counts else 0.0,
            "consensus_raw_loss": float(raw.detach().item()),
        }
        return raw, metrics

    @staticmethod
    def _slice_from_box(start: float, end: float, limit: int) -> tuple[int, int]:
        start_i = max(0, min(limit - 1, int(math.floor(start))))
        end_i = max(start_i + 1, min(limit, int(math.ceil(end))))
        return start_i, end_i

    @staticmethod
    def _dgfe_aux_list(preds: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        aux = preds.get("dgfe_aux")
        if aux is None:
            return []
        return aux if isinstance(aux, list) else [aux]

    def _dgfe_reconstruction_loss(self, preds: dict[str, Any], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        aux_list = self._dgfe_aux_list(preds)
        if not aux_list or "img" not in batch:
            return preds["boxes"].sum() * 0.0

        losses = []
        for aux in aux_list:
            recon = aux.get("recon")
            if recon is None:
                continue
            target = batch["img"].to(device=recon.device, dtype=recon.dtype)
            if target.shape[-2:] != recon.shape[-2:]:
                target = F.interpolate(target, size=recon.shape[-2:], mode="bilinear", align_corners=False)
            losses.append(F.smooth_l1_loss(recon, target))
        return torch.stack(losses).mean() if losses else preds["boxes"].sum() * 0.0

    def _dgfe_spatial_target(
        self,
        logits: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        q_by_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = torch.zeros_like(logits)
        batch_size, _, h, w = logits.shape
        image_h = max(float(imgsz[0]), 1.0)
        image_w = max(float(imgsz[1]), 1.0)
        ring = max(self.dgfe_boundary_ring, 0.0)
        inner_value = max(min(self.dgfe_inner_value, 1.0), 0.0)

        def write_max(region: torch.Tensor, value: float) -> None:
            if region.numel():
                region.copy_(torch.maximum(region, torch.full_like(region, value)))

        for bi in range(batch_size):
            valid_gt = mask_gt[bi, :, 0].bool()
            boxes = gt_bboxes[bi][valid_gt]
            quality = q_by_gt[bi][valid_gt].clamp(0, 1) if q_by_gt is not None else None
            for box_idx, box in enumerate(boxes):
                if quality is None:
                    q, left_value, top_value, right_value, bottom_value = 1.0, 1.0, 1.0, 1.0, 1.0
                else:
                    values = quality[box_idx]
                    if values.ndim == 0:
                        q = left_value = top_value = right_value = bottom_value = float(values)
                    else:
                        q, left_value, top_value, right_value, bottom_value = [float(v) for v in values[:5]]
                inside_value = inner_value * q
                boundary_value = max(left_value, top_value, right_value, bottom_value)
                x1, y1, x2, y2 = [float(v) for v in box]
                fx1, fx2 = x1 * w / image_w, x2 * w / image_w
                fy1, fy2 = y1 * h / image_h, y2 * h / image_h
                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                ix1, ix2 = self._slice_from_box(fx1, fx2, w)
                iy1, iy2 = self._slice_from_box(fy1, fy2, h)
                if (ix2 - ix1) * (iy2 - iy1) <= self.dgfe_tiny_area:
                    write_max(target[bi, :, iy1:iy2, ix1:ix2], boundary_value)
                    continue

                write_max(target[bi, :, iy1:iy2, ix1:ix2], inside_value)
                edge = max(int(math.ceil(ring)), 1)
                write_max(target[bi, :, iy1:min(iy1 + edge, iy2), ix1:ix2], top_value)
                write_max(target[bi, :, max(iy2 - edge, iy1):iy2, ix1:ix2], bottom_value)
                write_max(target[bi, :, iy1:iy2, ix1:min(ix1 + edge, ix2)], left_value)
                write_max(target[bi, :, iy1:iy2, max(ix2 - edge, ix1):ix2], right_value)

                ox1, ox2 = self._slice_from_box(fx1 - ring, fx2 + ring, w)
                oy1, oy2 = self._slice_from_box(fy1 - ring, fy2 + ring, h)
                write_max(target[bi, :, oy1:iy1, ox1:ox2], top_value)
                write_max(target[bi, :, iy2:oy2, ox1:ox2], bottom_value)
                write_max(target[bi, :, iy1:iy2, ox1:ix1], left_value)
                write_max(target[bi, :, iy1:iy2, ix2:ox2], right_value)
        return target

    def _dgfe_quality_by_gt(
        self,
        assigned_iou: torch.Tensor | None,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor | None:
        if assigned_iou is None or not fg_mask.any():
            return None
        quality = torch.zeros((*mask_gt.shape[:2], 5), device=mask_gt.device, dtype=assigned_iou.dtype)
        offset = 0
        for bi in range(fg_mask.shape[0]):
            fg = fg_mask[bi]
            n = int(fg.sum().item())
            if not n:
                continue
            gt_idx = target_gt_idx[bi, fg].long()
            gt_iou = assigned_iou.detach()[offset : offset + n]
            pred = pred_bboxes.detach()[bi, fg]
            target = target_bboxes.detach()[bi, fg]
            offset += n
            for gi in gt_idx.unique():
                group = gt_idx == gi
                best = gt_iou[group].argmax()
                q = gt_iou[group][best]
                side_values = q.repeat(4)
                if self.dgfe_spatial_target_mode == "edge_error":
                    pred_box = pred[group][best]
                    target_box = target[group][best]
                    w = (target_box[2] - target_box[0]).clamp_min(1e-9)
                    h = (target_box[3] - target_box[1]).clamp_min(1e-9)
                    side_error = torch.stack(
                        (
                            (pred_box[0] - target_box[0]).abs() / w,
                            (pred_box[1] - target_box[1]).abs() / h,
                            (pred_box[2] - target_box[2]).abs() / w,
                            (pred_box[3] - target_box[3]).abs() / h,
                        )
                    ).div(self.dgfe_edge_error_norm).clamp(0, 1)
                    side_values = torch.maximum(side_values, side_error)
                quality[bi, gi] = torch.cat((q.reshape(1), side_values))
        return quality

    def _dgfe_spatial_loss(
        self,
        preds: dict[str, Any],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        q_by_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aux_list = self._dgfe_aux_list(preds)
        if not aux_list:
            return preds["boxes"].sum() * 0.0

        losses = []
        for aux in aux_list:
            logits = aux.get("spatial_logits")
            if logits is None:
                continue
            target = self._dgfe_spatial_target(logits, gt_bboxes, mask_gt, imgsz, q_by_gt).to(dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            pos_mask = target > 0
            if not pos_mask.any():
                losses.append(logits.sum() * 0.0)
                continue

            pos_loss = bce[pos_mask].mean()
            neg_loss = logits.sum() * 0.0
            neg = bce[~pos_mask]
            if neg.numel():
                k = min(int(pos_mask.sum().item()) * max(self.dgfe_neg_pos_ratio, 1), neg.numel())
                neg_loss = neg.topk(k).values.mean()
            losses.append(pos_loss + self.dgfe_neg_gain * neg_loss)
        return torch.stack(losses).mean() if losses else preds["boxes"].sum() * 0.0

    def _dgfe_spatial_gain(self) -> float:
        if self.dgfe_spatial_warmup_epochs <= 0:
            return self.dgfe_spatial_gain
        t = min(max(float(self.epoch), 0.0) / max(float(self.dgfe_spatial_warmup_epochs), 1.0), 1.0)
        factor = self.dgfe_spatial_warmup_start + (1.0 - self.dgfe_spatial_warmup_start) * t
        return self.dgfe_spatial_gain * factor

    @staticmethod
    def _p2_detail_aux_list(preds: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        aux = preds.get("p2_detail_aux")
        if aux is None:
            return []
        return aux if isinstance(aux, list) else [aux]

    def _p2_detail_reconstruction_loss(self, preds: dict[str, Any]) -> torch.Tensor:
        aux_list = self._p2_detail_aux_list(preds)
        if not aux_list:
            return preds["boxes"].sum() * 0.0

        losses = []
        mask_fraction = []
        for aux in aux_list:
            pred, target, mask = aux.get("detail_pred"), aux.get("detail_target"), aux.get("detail_mask")
            if pred is None or target is None or mask is None:
                continue
            mask = mask.to(device=pred.device, dtype=pred.dtype)
            if mask.shape[-2:] != pred.shape[-2:]:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
            mask = mask.expand_as(pred)
            denom = mask.sum().clamp_min(1.0)
            target = target.to(device=pred.device, dtype=pred.dtype)
            losses.append(F.smooth_l1_loss(pred * mask, target * mask, reduction="sum") / denom)
            mask_fraction.append(mask[:, :1].mean().detach())
        raw = torch.stack(losses).mean() if losses else preds["boxes"].sum() * 0.0
        self.p2_detail_metrics = {
            "p2_detail_raw_loss": float(raw.detach().item()),
            "p2_detail_mask_fraction": float(torch.stack(mask_fraction).mean().item()) if mask_fraction else 0.0,
        }
        return raw

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(
            3
            + int(self.boundary_loss is not None)
            + int(self.loc_quality_loss is not None)
            + int(self.quality_head)
            + int(self.rank_gain > 0)
            + int(self.box_consensus_gain > 0)
            + int(self.psd_enabled)
            + int(self.dgfe_rec_gain > 0)
            + int(self.dgfe_spatial_gain > 0)
            + int(self.p2_detail_rec_gain > 0),
            device=self.device,
        )  # box, cls, dfl[, boundary][, loc_quality][, quality][, rank][, psd][, dgfe_rec][, dgfe_spatial][, p2_detail]
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        pred_residual = preds.get("dfl_residual")
        pred_residual = pred_residual.permute(0, 2, 1).contiguous() if pred_residual is not None else None
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_residual, stride_tensor)  # xyxy

        if self.assignment_override is None:
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        else:
            target_bboxes, target_scores, fg_mask, target_gt_idx = self.assignment_override
            self.assignment_override = None
            expected_shape = pred_scores.shape[:2]
            if target_bboxes.shape[:2] != expected_shape or target_scores.shape[:2] != expected_shape:
                raise ValueError("Reused TAL assignment does not match the auxiliary prediction shape.")
            target_bboxes = target_bboxes.to(device=self.device, dtype=gt_bboxes.dtype)
            target_scores = target_scores.to(device=self.device, dtype=pred_scores.dtype)
            fg_mask = fg_mask.to(device=self.device)
            target_gt_idx = target_gt_idx.to(device=self.device)
        self.support_assignment_metrics = dict(getattr(self.assigner, "last_metrics", {}))
        fg_mask = fg_mask.bool()
        flat_stride = stride_tensor.squeeze(-1)
        post_tal_support_metrics = self._post_tal_support_metrics(
            mask_gt, fg_mask, target_gt_idx, flat_stride, self.stride
        )
        self.last_assignment = (
            tuple(value.detach().clone() for value in (target_bboxes, target_scores, fg_mask, target_gt_idx))
            if self.capture_assignment
            else None
        )
        n_p2 = math.prod(preds["feats"][0].shape[2:])
        self.dbss_assignment_context = {
            "p2_fg_mask": fg_mask[:, :n_p2].detach(),
            "p2_target_scores": target_scores[:, :n_p2].detach(),
            "p2_dfl_conflict": fg_mask[:, :n_p2].new_zeros(
                (batch_size, *preds["feats"][0].shape[-2:]), dtype=pred_distri.dtype
            ),
            "total_positive_count": fg_mask.sum().detach(),
        }

        # Positive-Support Dropout (Dominant-Support Suppression) Mask building
        h2, w2 = preds["feats"][0].shape[-2:]
        drop_mask = torch.zeros((batch_size, 1, h2, w2), device=self.device, dtype=dtype)
        aux_fg_mask = torch.zeros((batch_size, n_p2), device=self.device, dtype=torch.bool)
        
        # Diagnostic metrics variables
        total_eligible_gt = 0
        total_alt_supports_found = 0
        
        if self.psd_enabled and fg_mask[:, :n_p2].any():
            p2_fg = fg_mask[:, :n_p2]
            p2_gt_idx = target_gt_idx[:, :n_p2]
            p2_scores = target_scores[:, :n_p2].sum(-1)  # alignment quality sum (classes)
            
            for b in range(batch_size):
                valid_gt_num = int(mask_gt[b].sum().item())
                for g in range(valid_gt_num):
                    p_g_mask = p2_fg[b] & (p2_gt_idx[b] == g)
                    count = int(p_g_mask.sum().item())
                    if count < self.psd_min_count:
                        continue
                    
                    # Find dominant support
                    g_scores = torch.where(p_g_mask, p2_scores[b], p2_scores.new_full(p2_scores[b].shape, -1.0))
                    i_g_star = int(g_scores.argmax().item())
                    
                    # Build list of valid centers that have at least one alternative support within radius
                    pos_indices = p_g_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                    valid_centers = {}
                    for i in pos_indices:
                        y_i, x_i = i // w2, i % w2
                        alts = []
                        for j in pos_indices:
                            if i == j:
                                continue
                            y_j, x_j = j // w2, j % w2
                            if max(abs(y_j - y_i), abs(x_j - x_i)) <= self.psd_radius:
                                alts.append(j)
                        if alts:
                            valid_centers[i] = alts
                            
                    if not valid_centers:
                        continue
                        
                    # Select i_center
                    i_center = None
                    if self.psd_mode == "dominant":
                        if i_g_star in valid_centers:
                            i_center = i_g_star
                    elif self.psd_mode == "random":
                        keys = list(valid_centers.keys())
                        idx_rand = int(torch.randint(0, len(keys), (1,)).item())
                        i_center = keys[idx_rand]
                    elif self.psd_mode == "none":
                        if i_g_star in valid_centers:
                            i_center = i_g_star
                            
                    if i_center is None:
                        continue
                        
                    # Decide on intervention with probability self.psd_prob (or run matches)
                    do_drop = (float(torch.rand(1).item()) < self.psd_prob) or (self.psd_mode == "none")
                    if do_drop:
                        total_eligible_gt += 1
                        
                        if self.psd_mode != "none":
                            y_mask, x_mask = i_center // w2, i_center % w2
                            drop_mask[b, 0, y_mask, x_mask] = 1.0
                            
                        # Set alternative supports in aux_fg_mask
                        alt_candidates = valid_centers[i_center]
                        cand_tensor = torch.tensor(alt_candidates, device=self.device, dtype=torch.long)
                        cand_scores = p2_scores[b, cand_tensor]
                        topk_num = min(self.psd_aux_topk, len(alt_candidates))
                        topk_indices = cand_tensor[cand_scores.topk(topk_num).indices]
                        
                        aux_fg_mask[b, topk_indices] = True
                        total_alt_supports_found += topk_num

        target_scores_sum = max(target_scores.sum(), 1)
        # TAL ends above. Everything below consumes its assignment without
        # changing target_scores, fg_mask, target_gt_idx, or positive count.
        target_bboxes_scaled = target_bboxes / stride_tensor
        realized_iou = None
        if self.ftsc_calibrator is not None and fg_mask.any():
            realized_iou = bbox_iou(
                pred_bboxes.detach()[fg_mask],
                target_bboxes_scaled[fg_mask],
                xywh=False,
                CIoU=False,
            ).squeeze(-1).clamp(0, 1).detach()
        ftsc_weights = None
        self.ftsc_metrics = dict(self.support_assignment_metrics)
        self.ftsc_metrics.update(post_tal_support_metrics)
        if fg_mask.any():
            assigned_scores = target_scores.sum(-1)[fg_mask].detach().float()
            self.ftsc_metrics.update(
                {
                    "support_assigned_target_score_mean": float(assigned_scores.mean().item()),
                    "support_assigned_target_score_std": float(assigned_scores.std(unbiased=False).item()),
                    "support_assigned_realized_iou_mean": float(realized_iou.mean().item()) if realized_iou is not None else 0.0,
                    "support_assigned_realized_iou_std": float(realized_iou.std(unbiased=False).item()) if realized_iou is not None else 0.0,
                }
            )
        # YOLODataset.collate_fn keeps arbitrary per-sample metadata as a tuple;
        # accept both tuple and list so the existing AdaptiveZoom diagnostics
        # survive collation into the training batch.
        zoom_records = batch.get("adaptive_zoom")
        if isinstance(zoom_records, (list, tuple)) and zoom_records and isinstance(zoom_records[0], dict):
            count = float(len(zoom_records))
            self.ftsc_metrics.update(
                {
                    "zoom_applied_fraction": sum(float(item.get("applied", 0.0)) for item in zoom_records) / count,
                    "zoom_factor_mean": sum(float(item.get("factor", 1.0)) for item in zoom_records) / count,
                    "zoom_factor_std": max(
                        sum(float(item.get("factor_sq", 1.0)) for item in zoom_records) / count
                        - (sum(float(item.get("factor", 1.0)) for item in zoom_records) / count) ** 2,
                        0.0,
                    ) ** 0.5,
                    "zoom_factor_max": max(float(item.get("factor", 1.0)) for item in zoom_records),
                    "tiny_gt_count": sum(float(item.get("tiny_gt_count", 0.0)) for item in zoom_records),
                    "zoomed_gt_count": sum(float(item.get("zoomed_gt_count", 0.0)) for item in zoom_records),
                    "boxes_dropped_by_zoom": sum(float(item.get("boxes_dropped", 0.0)) for item in zoom_records),
                    "empty_after_zoom_count": sum(float(item.get("empty_after", 0.0)) for item in zoom_records),
                    "bbox_min_side_before_mean": sum(float(item.get("bbox_min_side_before", 0.0)) for item in zoom_records) / count,
                    "bbox_min_side_after_mean": sum(float(item.get("bbox_min_side_after", 0.0)) for item in zoom_records) / count,
                }
            )
        if self.ftsc_calibrator is not None:
            classification_quality = None
            if "dcfl_dgmm_quality" in self.ftsc_calibrator.evidence_names and fg_mask.any():
                with torch.no_grad():
                    positive_classes = target_scores[fg_mask].argmax(-1, keepdim=True)
                    classification_quality = pred_scores.detach().sigmoid()[fg_mask].gather(
                        -1, positive_classes
                    ).squeeze(-1)
            ftsc_kwargs = {
                "epoch": self.epoch,
                "classification_quality": classification_quality,
                "localization_quality": realized_iou,
            }
            if self.ftsc_calibrator.audit_enabled:
                ftsc_kwargs.update(
                    {
                        "audit_target_scores": target_scores,
                        "audit_stride": flat_stride,
                        "audit_anchor_index": torch.arange(
                            fg_mask.shape[1], device=fg_mask.device, dtype=torch.long
                        ).view(1, -1).expand_as(fg_mask),
                    }
                )
            ftsc_weights = self.ftsc_calibrator(
                anchor_points * stride_tensor,
                target_bboxes,
                target_gt_idx,
                fg_mask,
                pred_distri,
                **ftsc_kwargs,
            )
            self.ftsc_metrics.update(self.ftsc_calibrator.last_metrics)
            target_score_dense = target_scores.sum(-1)
            realized_iou_dense = torch.zeros_like(target_score_dense, dtype=torch.float32)
            if realized_iou is not None and fg_mask.any():
                realized_iou_dense[fg_mask] = realized_iou.float()
            # These diagnostics deliberately derive from the existing
            # assignment/prediction tensors; they do not perform another
            # forward pass and remain valid for any number of Detect levels.
            # Keep the unclipped distance for the range diagnostic.  The loss
            # path clips to ``reg_max - 1 - 0.01``; using that clipped tensor
            # here would make the requested near-limit fraction identically
            # zero for the threshold ``reg_max - 1.5``.
            target_ltrb_raw_dense = bbox2dist(anchor_points, target_bboxes_scaled, None).clamp_min(0)
            target_ltrb_dense = target_ltrb_raw_dense.clamp_max(self.reg_max - 1 - 0.01)
            dfl_probability = pred_distri.view(pred_distri.shape[0], pred_distri.shape[1], 4, self.reg_max).softmax(-1)
            dfl_entropy_dense = -(dfl_probability * dfl_probability.clamp_min(1e-9).log()).sum(-1)
            for level_stride in self.stride.tolist():
                level_mask = flat_stride == float(level_stride)
                level_positive = fg_mask & level_mask.unsqueeze(0)
                label = _level_label(float(level_stride))
                prefix = f"ftsc_level_{label}"
                count = int(level_positive.sum().item())
                self.ftsc_metrics[f"{prefix}_positive_count"] = float(count)
                self.ftsc_metrics[f"{prefix}_positive_fraction"] = float(count / max(float(fg_mask.sum().item()), 1.0))
                self.ftsc_metrics[f"{prefix}_target_score_mean"] = float(
                    target_score_dense[level_positive].mean().item() if count else 0.0
                )
                self.ftsc_metrics[f"{prefix}_target_score_mass"] = float(target_score_dense[level_positive].sum().item())
                self.ftsc_metrics[f"{prefix}_realized_iou_mean"] = float(
                    realized_iou_dense[level_positive].mean().item() if count else 0.0
                )
                self.ftsc_metrics[f"{prefix}_dfl_entropy_mean"] = float(
                    dfl_entropy_dense[level_positive].mean().item() if count else 0.0
                )
                if ftsc_weights is not None:
                    positive_selector = level_positive[fg_mask]
                    for weight_name in ("cls", "box", "dfl"):
                        values = ftsc_weights[weight_name][positive_selector].detach().float()
                        self.ftsc_metrics[f"{prefix}_ftsc_{weight_name}_weight_mean"] = float(
                            values.mean().item() if values.numel() else 0.0
                        )
                        self.ftsc_metrics[f"{prefix}_ftsc_{weight_name}_weight_std"] = float(
                            values.std(unbiased=False).item() if values.numel() else 0.0
                        )
                target_values = target_ltrb_dense[level_positive]
                target_values_raw = target_ltrb_raw_dense[level_positive]
                if target_values.numel():
                    self.ftsc_metrics[f"{prefix}_target_ltrb_max_bin_mean"] = float(target_values_raw.max(-1).values.mean().item())
                    self.ftsc_metrics[f"{prefix}_dfl_near_regmax_fraction"] = float(
                        (target_values_raw >= self.reg_max - 1.5).float().mean().item()
                    )
                else:
                    self.ftsc_metrics[f"{prefix}_target_ltrb_max_bin_mean"] = 0.0
                    self.ftsc_metrics[f"{prefix}_dfl_near_regmax_fraction"] = 0.0
                # Keep the compact historical names for downstream scripts.
                self.ftsc_metrics[f"ftsc_positive_count_{label}"] = float(count)
                self.ftsc_metrics[f"ftsc_positive_fraction_{label}"] = float(count / max(float(fg_mask.sum().item()), 1.0))
            if fg_mask.any():
                positive_scores = target_scores.sum(-1)[fg_mask]
                positive_boxes = target_bboxes[fg_mask]
                positive_area = (
                    (positive_boxes[:, 2] - positive_boxes[:, 0]).clamp_min(0)
                    * (positive_boxes[:, 3] - positive_boxes[:, 1]).clamp_min(0)
                )
                cls_ftsc_weight = ftsc_weights["cls"]
                self.ftsc_metrics["ftsc_corr_weight_target_score"] = self._ftsc_correlation(
                    cls_ftsc_weight, positive_scores
                )
                self.ftsc_metrics["ftsc_corr_weight_gt_area"] = self._ftsc_correlation(
                    cls_ftsc_weight, positive_area
                )
                if "dcfl_dgmm_quality" in self.ftsc_calibrator.providers:
                    dcfl_pt = self.ftsc_calibrator.providers["dcfl_dgmm_quality"].last_pt
                    self.ftsc_metrics["ftsc_dcfl_corr_weight_pt"] = self._ftsc_correlation(
                        cls_ftsc_weight, dcfl_pt
                    )
                batch_ids = torch.arange(batch_size, device=self.device).view(-1, 1).expand_as(fg_mask)[fg_mask]
                group_ids = batch_ids * fg_mask.shape[1] + target_gt_idx[fg_mask].long()
                _, group_counts = group_ids.unique(return_counts=True)
                group_weight_sum = cls_ftsc_weight.new_zeros(batch_size * fg_mask.shape[1]).scatter_add_(
                    0, group_ids, cls_ftsc_weight
                )
                group_count_dense = cls_ftsc_weight.new_zeros(batch_size * fg_mask.shape[1]).scatter_add_(
                    0, group_ids, torch.ones_like(cls_ftsc_weight)
                )
                group_weight_means = group_weight_sum[group_count_dense > 0] / group_count_dense[
                    group_count_dense > 0
                ]
                self.ftsc_metrics["ftsc_mean_positives_per_gt"] = float(group_counts.float().mean().item())
                self.ftsc_metrics["ftsc_single_positive_gt_fraction"] = float(
                    (group_counts == 1).float().mean().item()
                )
                self.ftsc_metrics["ftsc_per_gt_weight_mean"] = float(group_weight_means.float().mean().item())
                self.ftsc_metrics["ftsc_per_gt_weight_mean_std"] = float(
                    group_weight_means.float().std(unbiased=False).item()
                )
        cls_target_scores = target_scores
        cls_target_scores_sum = target_scores_sum
        assigned_iou = None
        if ftsc_weights is not None and realized_iou is not None:
            self.ftsc_metrics["ftsc_corr_weight_realized_iou"] = self._ftsc_correlation(
                ftsc_weights["cls"], realized_iou
            )
        loc_target_bboxes, loc_target_scores, loc_fg_mask = self.build_localization_targets(
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            target_bboxes,
            target_scores,
            fg_mask,
        )
        loc_target_bboxes_scaled = loc_target_bboxes / stride_tensor
        loc_target_scores_sum = max(loc_target_scores.sum(), 1)

        # Cls loss with optional class weighting
        if (self.vfl is not None or getattr(self.hyp, "cls_iou_target", False)) and fg_mask.sum():
            iou_pred_bboxes = pred_bboxes[fg_mask]
            if self.vfl is None or bool(getattr(self.hyp, "vfl_iou_detach", True)):
                iou_pred_bboxes = iou_pred_bboxes.detach()
            assigned_iou = bbox_iou(
                iou_pred_bboxes,
                target_bboxes_scaled[fg_mask],
                xywh=False,
                CIoU=False,
            ).squeeze(-1).clamp(0)
        # Compute size-aware classification weights
        cls_weights = torch.ones_like(cls_target_scores)
        import os
        variant = os.environ.get("YOLO_VARIANT", "")
        if "small_weight" in variant and fg_mask.sum():
            if "preclip_area" in batch:
                ori_targets = torch.cat((
                    batch["batch_idx"].view(-1, 1), 
                    batch["cls"].view(-1, 1), 
                    batch["bboxes"],
                    batch["preclip_area"].view(-1, 1)
                ), 1)
                ori_targets = self.preprocess(ori_targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                gt_preclip_area = ori_targets[..., 5]
                
                for b in range(batch_size):
                    fg_b = fg_mask[b]
                    if fg_b.any():
                        gt_idx = target_gt_idx[b, fg_b]
                        A_g = gt_preclip_area[b, gt_idx]
                        
                        w_g = torch.ones_like(A_g)
                        w_g = torch.where(A_g < 100.0, torch.full_like(w_g, 1.75), w_g)
                        w_g = torch.where((A_g >= 100.0) & (A_g < 400.0), torch.full_like(w_g, 1.50), w_g)
                        w_g = torch.where((A_g >= 400.0) & (A_g < 1000.0), torch.full_like(w_g, 1.15), w_g)
                        
                        alpha_t = min(max(float(getattr(self, "epoch", 0.0)), 0.0) / 5.0, 1.0)
                        w_g = 1.0 + alpha_t * (w_g - 1.0)
                        
                        fg_indices = torch.nonzero(fg_b).squeeze(-1)
                        classes_b = target_scores[b, fg_b].argmax(-1).long()
                        cls_weights[b, fg_indices, classes_b] = w_g

        if ftsc_weights is not None and fg_mask.any():
            # Only target-class positive BCE/VFL elements are calibrated. Background
            # anchors and off-class elements at a positive anchor remain at identity.
            cls_weights = self._apply_ftsc_positive_cls_weights(
                cls_weights,
                target_scores,
                fg_mask,
                ftsc_weights["cls"],
            )

        if self.vfl is not None:
            if assigned_iou is not None:
                cls_target_scores = target_scores.clone()
                cls_target_scores[fg_mask] = torch.where(
                    target_scores[fg_mask] > 0,
                    assigned_iou.unsqueeze(-1).to(dtype=target_scores.dtype),
                    target_scores[fg_mask],
                )
            cls_target_scores_sum = max(cls_target_scores.sum(), 1)
            cls_labels = (cls_target_scores > 0).to(dtype)
            loss[1] = self.vfl(pred_scores, cls_target_scores.to(dtype), cls_labels, cls_weights) / cls_target_scores_sum
        elif getattr(self.hyp, "cls_iou_target", False) and assigned_iou is not None:
            with torch.no_grad():
                cls_target_scores = target_scores.clone()
                cls_target_scores[fg_mask] = torch.where(
                    target_scores[fg_mask] > 0,
                    assigned_iou.unsqueeze(-1),
                    target_scores[fg_mask],
                )
                cls_target_scores_sum = max(cls_target_scores.sum(), 1)
            bce_loss = self.bce(pred_scores, cls_target_scores.to(dtype))  # (bs, num_anchors, nc)
            bce_loss *= cls_weights
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / cls_target_scores_sum  # BCE
        elif self.ghm_cls_loss is not None:
            # GHM-C is a full dense classification objective, including
            # negatives. It does not use FTSC's positive-only evidence
            # weights, but it keeps YOLO's target-score normalization so the
            # classification-loss scale remains comparable to the BCE control.
            loss[1] = self.ghm_cls_loss(
                pred_scores,
                cls_target_scores.to(dtype),
                cls_target_scores_sum,
                self.class_weights,
            )
        else:
            bce_loss = self.bce(pred_scores, cls_target_scores.to(dtype))  # (bs, num_anchors, nc)
            if (
                self.ftsc_calibrator is not None
                and self.ftsc_calibrator.mal_loss is not None
                and realized_iou is not None
            ):
                positive_target_mask = target_scores > 0
                mal_quality = realized_iou[target_scores[fg_mask].amax(-1) > 0]
                bce_loss, positive_mal = self.ftsc_calibrator.mal_loss.replace_dense(
                    bce_loss,
                    pred_scores,
                    positive_target_mask,
                    mal_quality,
                )
                positive_confidence = pred_scores[positive_target_mask].detach().sigmoid()
                quality = mal_quality.detach().float()
                self.ftsc_metrics.update(
                    {
                        "ftsc_mal_gamma": self.ftsc_calibrator.mal_gamma,
                        "ftsc_mal_positive_count": float(positive_mal.numel()),
                        "ftsc_mal_positive_loss_mean": (
                            float(positive_mal.detach().float().mean().item()) if positive_mal.numel() else 0.0
                        ),
                        "ftsc_mal_corr_q_confidence": self._ftsc_correlation(
                            mal_quality, positive_confidence
                        ),
                        "ftsc_mal_q_mean": float(quality.mean().item()) if quality.numel() else 0.0,
                        "ftsc_mal_q_std": (
                            float(quality.std(unbiased=False).item()) if quality.numel() else 0.0
                        ),
                        "ftsc_mal_q_min": float(quality.min().item()) if quality.numel() else 0.0,
                        "ftsc_mal_q_max": float(quality.max().item()) if quality.numel() else 0.0,
                    }
                )
            bce_loss *= cls_weights
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / cls_target_scores_sum  # BCE

        self.positive_confidence_rescue_metrics = {}
        if self.positive_confidence_rescue_gain > 0:
            raw_rescue, target, logits = self.positive_confidence_rescue_loss(pred_scores, target_scores, fg_mask)
            original_cls = loss[1].detach()
            applied_rescue = raw_rescue * self.positive_confidence_rescue_gain
            loss[1] = loss[1] + applied_rescue
            target_quantiles = (
                torch.quantile(target.float(), torch.tensor([0.25, 0.5, 0.75], device=target.device))
                if target.numel()
                else target.new_zeros(3).float()
            )
            scores = logits.detach().float().sigmoid()
            self.positive_confidence_rescue_metrics = {
                "rescue_positive_count": float(target.numel()),
                "rescue_target_mean": float(target.float().mean().item()) if target.numel() else 0.0,
                "rescue_target_q25": float(target_quantiles[0].item()),
                "rescue_target_q50": float(target_quantiles[1].item()),
                "rescue_target_q75": float(target_quantiles[2].item()),
                "rescue_positive_score_mean": float(scores.mean().item()) if scores.numel() else 0.0,
                "rescue_score_target_lt_025": float(scores[target < 0.25].mean().item()) if (target < 0.25).any() else 0.0,
                "rescue_score_target_025_050": float(scores[(target >= 0.25) & (target < 0.5)].mean().item())
                if ((target >= 0.25) & (target < 0.5)).any() else 0.0,
                "rescue_score_target_ge_050": float(scores[target >= 0.5].mean().item())
                if (target >= 0.5).any() else 0.0,
                "rescue_raw_loss": float(raw_rescue.detach().item()),
                "rescue_applied_loss": float(applied_rescue.detach().item()),
                "rescue_to_cls_loss_ratio": float(applied_rescue.detach().div(original_cls.clamp_min(1e-12)).item()),
            }

        # Bbox loss. Optionally use a localization-only positive set that does not affect cls targets.
        self.bbox_loss.last_p2_conflict = None
        self.bbox_loss.last_pc_loss = None
        if loc_fg_mask.sum():
            quality_weights = None
            if bool(getattr(self.hyp, "vfl_weight_box_by_q", False)):
                with torch.no_grad():
                    box_assigned_iou = bbox_iou(
                        pred_bboxes.detach()[loc_fg_mask],
                        loc_target_bboxes_scaled[loc_fg_mask],
                        xywh=False,
                        CIoU=False,
                    ).squeeze(-1).clamp(0)
                quality_weights = box_assigned_iou.detach().clamp_min(float(getattr(self.hyp, "box_q_weight_min", 0.25)))
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                loc_target_bboxes_scaled,
                loc_target_scores,
                loc_target_scores_sum,
                loc_fg_mask,
                imgsz,
                stride_tensor,
                quality_weights,
                ftsc_weights["box"] if ftsc_weights is not None else None,
                ftsc_weights["dfl"] if ftsc_weights is not None else None,
            )
            if self.bbox_loss.last_p2_conflict is not None:
                h2, w2 = preds["feats"][0].shape[-2:]
                self.dbss_assignment_context["p2_dfl_conflict"] = self.bbox_loss.last_p2_conflict[:, :n_p2].reshape(
                    batch_size, h2, w2
                )
                self.dbss_assignment_context["loss_pc_dfl"] = self.bbox_loss.last_pc_loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if ftsc_weights is not None:
            ftsc_regularization = ftsc_weights["regularization"]
            loss[1] = loss[1] + ftsc_regularization
            self.ftsc_metrics["ftsc_strength_regularization"] = float(ftsc_regularization.detach().item())
            if self.ftsc_calibrator.um_enabled:
                weighted_um = self.ftsc_calibrator.um_lambda * ftsc_weights["um_loss"]
                loss[2] = loss[2] + weighted_um
                self.ftsc_metrics["ftsc_um_weighted_loss"] = float(weighted_um.detach().item())
        if self.boundary_loss is not None:
            # EXPERIMENTAL: add feature-space boundary separation only during training loss computation.
            loss[3] = self.boundary_loss(preds["feats"], gt_bboxes, mask_gt, self.stride) * self.boundary_gain
        if self.loc_quality_loss is not None:
            # EXPERIMENTAL: train-only smooth localization map supervision.
            loc_idx = 3 + int(self.boundary_loss is not None)
            loc_maps = preds.get("loc_maps")
            if loc_maps is None:
                loc_maps = [self.loc_quality_heads[i](x) for i, x in enumerate(preds["feats"])]
            loss[loc_idx] = self.loc_quality_loss(loc_maps, gt_bboxes, mask_gt, self.stride) * self.loc_quality_gain
        if self.quality_head:
            quality_idx = 3 + int(self.boundary_loss is not None) + int(self.loc_quality_loss is not None)
            quality_logits = preds.get("quality_logits")
            if quality_logits is None:
                raise KeyError("quality_head=True but model predictions do not contain 'quality_logits'.")
            quality_logits = quality_logits.permute(0, 2, 1).contiguous().squeeze(-1)
            quality_target = torch.zeros_like(quality_logits)
            max_iou_to_gt = torch.zeros_like(quality_logits)
            pred_bboxes_for_quality = pred_bboxes.detach() if self.quality_detach_target else pred_bboxes
            for bi in range(batch_size):
                valid_gt = mask_gt[bi].squeeze(-1).bool()
                if valid_gt.any():
                    pred_pixel = pred_bboxes_for_quality[bi] * stride_tensor
                    max_iou_to_gt[bi] = box_iou(pred_pixel, gt_bboxes[bi, valid_gt]).clamp(0).amax(-1)
            if fg_mask.sum():
                quality_iou = bbox_iou(
                    pred_bboxes.detach()[fg_mask] if self.quality_detach_target else pred_bboxes[fg_mask],
                    target_bboxes_scaled[fg_mask],
                    xywh=False,
                    CIoU=False,
                ).squeeze(-1).clamp(0)
                if self.quality_detach_target:
                    quality_iou = quality_iou.detach()
                quality_target[fg_mask] = quality_iou.to(dtype=quality_target.dtype)
            if self.quality_loss_type == "bce_balanced":
                if self.quality_detach_target:
                    max_iou_to_gt = max_iou_to_gt.detach()
                q_pos_mask = fg_mask | (max_iou_to_gt > self.quality_pos_iou_thr)
                q_target_value = max_iou_to_gt.clamp(0, 1)
                if self.quality_target_mode == "iou_power":
                    q_target_value = q_target_value.pow(self.quality_target_power)
                else:
                    q_target_value = ((q_target_value - self.quality_ramp_low) / (
                        self.quality_ramp_high - self.quality_ramp_low
                    )).clamp(0, 1)
                if self.quality_detach_target:
                    q_target_value = q_target_value.detach()
                quality_target[q_pos_mask] = q_target_value[q_pos_mask].to(dtype=quality_target.dtype)
                pos_loss = (
                    F.binary_cross_entropy_with_logits(
                        quality_logits[q_pos_mask],
                        quality_target[q_pos_mask],
                        reduction="mean",
                    )
                    if q_pos_mask.any()
                    else quality_logits.sum() * 0.0
                )
                if self.quality_neg_mode == "hard":
                    cls_score = pred_scores.detach().sigmoid().amax(-1)
                    neg_mask = (
                        (~q_pos_mask)
                        & (max_iou_to_gt < self.quality_hard_neg_iou_thr)
                        & (cls_score > self.quality_hard_neg_score_thr)
                    )
                else:
                    neg_mask = ~q_pos_mask
                neg_loss = (
                    F.binary_cross_entropy_with_logits(
                        quality_logits[neg_mask],
                        torch.zeros_like(quality_logits[neg_mask]),
                        reduction="mean",
                    )
                    if neg_mask.any()
                    else quality_logits.sum() * 0.0
                )
                loss[quality_idx] = self.quality_gain * (pos_loss + self.quality_neg_gain * neg_loss)
                debug_pos_loss, debug_neg_loss = pos_loss, neg_loss
                debug_q_pos_mask, debug_neg_mask = q_pos_mask, neg_mask
            elif self.quality_loss_type == "l1":
                pos_loss = (
                    F.l1_loss(quality_logits.sigmoid()[fg_mask], quality_target[fg_mask], reduction="sum")
                    if fg_mask.sum()
                    else quality_logits.sum() * 0.0
                )
                neg_loss = self.bce(quality_logits, quality_target).masked_fill(fg_mask, 0).sum()
                loss[quality_idx] = (pos_loss + neg_loss) / fg_mask.sum().clamp_min(1)
                debug_pos_loss, debug_neg_loss = pos_loss, neg_loss
                debug_q_pos_mask, debug_neg_mask = fg_mask, ~fg_mask
            else:
                loss[quality_idx] = self.bce(quality_logits, quality_target).sum() / fg_mask.sum().clamp_min(1)
                debug_pos_loss = loss[quality_idx]
                debug_neg_loss = quality_logits.sum() * 0.0
                debug_q_pos_mask, debug_neg_mask = fg_mask, ~fg_mask
            if self.quality_debug and self.quality_debug_seen < self.quality_debug_batches:
                self.quality_debug_seen += 1
                num_tal_pos = int(fg_mask.sum().item())
                num_q_pos = int(debug_q_pos_mask.sum().item())
                num_neg = int(debug_neg_mask.sum().item())
                if debug_q_pos_mask.any():
                    q_iou_dbg = quality_target[debug_q_pos_mask].detach().float()
                    iou_stats = (
                        float(q_iou_dbg.min().item()),
                        float(q_iou_dbg.mean().item()),
                        float(q_iou_dbg.max().item()),
                    )
                    max_iou_dbg = max_iou_to_gt[debug_q_pos_mask].detach().float()
                    max_iou_stats = (
                        float(max_iou_dbg.min().item()),
                        float(max_iou_dbg.mean().item()),
                        float(max_iou_dbg.max().item()),
                    )
                    pred_sample = pred_bboxes.detach()[debug_q_pos_mask][:5].float().cpu().tolist()
                    target_sample = quality_target.detach()[debug_q_pos_mask][:5].float().cpu().tolist()
                else:
                    iou_stats = (float("nan"), float("nan"), float("nan"))
                    max_iou_stats = (float("nan"), float("nan"), float("nan"))
                    pred_sample, target_sample = [], []
                LOGGER.info(
                    "quality_debug batch=%s/%s num_tal_pos=%s num_q_pos=%s num_neg=%s "
                    "iou_min/mean/max=(%.6g, %.6g, %.6g) max_iou_min/mean/max=(%.6g, %.6g, %.6g) "
                    "pos_loss=%.6g neg_loss=%.6g quality_gain=%.6g quality_neg_gain=%.6g loss_q=%.6g "
                    "quality_neg_mode=%s pos_thr=%.6g hard_neg_iou_thr=%.6g hard_neg_score_thr=%.6g "
                    "target_mode=%s target_power=%.6g ramp_low=%.6g ramp_high=%.6g",
                    self.quality_debug_seen,
                    self.quality_debug_batches,
                    num_tal_pos,
                    num_q_pos,
                    num_neg,
                    *iou_stats,
                    *max_iou_stats,
                    float(debug_pos_loss.detach().item()),
                    float(debug_neg_loss.detach().item()),
                    self.quality_gain,
                    self.quality_neg_gain,
                    float(loss[quality_idx].detach().item()),
                    self.quality_neg_mode,
                    self.quality_pos_iou_thr,
                    self.quality_hard_neg_iou_thr,
                    self.quality_hard_neg_score_thr,
                    self.quality_target_mode,
                    self.quality_target_power,
                    self.quality_ramp_low,
                    self.quality_ramp_high,
                )
                LOGGER.info("quality_debug pred_bboxes_sample=%s", pred_sample)
                LOGGER.info("quality_debug quality_target_sample=%s", target_sample)
        if self.rank_gain > 0:
            rank_idx = (
                3 + int(self.boundary_loss is not None) + int(self.loc_quality_loss is not None) + int(self.quality_head)
            )
            loss[rank_idx] = (
                self.pairwise_ranking_loss(
                    pred_scores,
                    pred_bboxes,
                    target_bboxes_scaled,
                    target_scores,
                    target_gt_idx,
                    fg_mask,
                )
                * self.rank_gain
            )
        consensus_idx = (
            3
            + int(self.boundary_loss is not None)
            + int(self.loc_quality_loss is not None)
            + int(self.quality_head)
            + int(self.rank_gain > 0)
        )
        self.consensus_metrics = {}
        if self.box_consensus_gain > 0:
            raw_consensus, self.consensus_metrics = self.box_consensus_loss(
                pred_bboxes, target_bboxes_scaled, target_gt_idx, fg_mask, n_p2, mask_gt
            )
            current_gain = self._box_consensus_current_gain()
            loss[consensus_idx] = raw_consensus * current_gain
            self.consensus_metrics.update(
                consensus_gain=current_gain,
                consensus_applied_loss=float(loss[consensus_idx].detach().item()),
                consensus_to_box_loss_ratio=float(
                    loss[consensus_idx].detach().div(loss[0].detach().clamp_min(1e-12)).item()
                ),
            )
            grad_ratio = 0.0
            if self.box_consensus_log_grad_ratio and current_gain > 0 and raw_consensus.requires_grad:
                cons_grad = torch.autograd.grad(loss[consensus_idx], pred_bboxes, retain_graph=True)[0]
                box_grad = torch.autograd.grad(loss[0], pred_bboxes, retain_graph=True)[0]
                grad_ratio = float(
                    cons_grad.detach().norm().div(box_grad.detach().norm().clamp_min(1e-12)).item()
                )
            self.consensus_metrics["consensus_to_box_grad_ratio"] = grad_ratio
        # Auxiliary pass for Positive-Support Dropout (Dominant-Support Suppression)
        psd_loss = torch.zeros(1, device=self.device, dtype=dtype)
        num_gt_losses = 0
        total_iou_main = 0.0
        total_iou_aux = 0.0
        total_iou_count = 0
        
        if self.psd_enabled:
            # Create auxiliary feature view
            p2 = preds["feats"][0]
            
            # Border-safe local ring mean (excluding center)
            ones = torch.ones_like(p2)
            weight = torch.ones((p2.shape[1], 1, 3, 3), device=self.device, dtype=p2.dtype)
            sum_val = F.conv2d(p2, weight=weight, padding=1, groups=p2.shape[1])
            count_val = F.conv2d(ones, weight=weight, padding=1, groups=p2.shape[1])
            local_mean = (sum_val - p2) / (count_val - 1.0).clamp_min(1.0)
            local_mean = local_mean.detach()
            
            p2_aux = p2 * (1 - drop_mask) + local_mean * drop_mask
            
            # Temporary BatchNorm settings to prevent running stats updates
            bn_states = []
            for m_sub in self.detect_head.cv2[0].modules():
                if isinstance(m_sub, nn.BatchNorm2d):
                    bn_states.append((m_sub, m_sub.track_running_stats, m_sub.training))
                    m_sub.track_running_stats = False
                    m_sub.train(True)
            try:
                aux_logits_map = self.detect_head.cv2[0](p2_aux)
            finally:
                for m_sub, track_stats, training in bn_states:
                    m_sub.track_running_stats = track_stats
                    m_sub.train(training)
            
            # Flatten and decode
            aux_logits = aux_logits_map.permute(0, 2, 3, 1).reshape(batch_size, n_p2, -1)
            aux_bboxes = self.bbox_decode(
                anchor_points[:n_p2], 
                aux_logits, 
                pred_residual[:, :n_p2] if pred_residual is not None else None, 
                stride_tensor[:n_p2] if stride_tensor is not None else None
            )
            
            # Calculate loss per GT on alternative supports
            if aux_fg_mask.any():
                p2_fg = fg_mask[:, :n_p2]
                p2_gt_idx = target_gt_idx[:, :n_p2]
                for b in range(batch_size):
                    valid_gt_num = int(mask_gt[b].sum().item())
                    for g in range(valid_gt_num):
                        p_g_alt = aux_fg_mask[b] & p2_fg[b] & (p2_gt_idx[b] == g)
                        if not p_g_alt.any():
                            continue
                        
                        pred_box_alt = aux_bboxes[b, p_g_alt]
                        target_box_alt = target_bboxes_scaled[b, :n_p2][p_g_alt]
                        
                        # CIoU
                        iou_aux = bbox_iou(pred_box_alt, target_box_alt, xywh=False, CIoU=True)
                        loss_iou = 1.0 - iou_aux.squeeze(-1)
                        
                        # DFL loss
                        target_ltrb_alt = bbox2dist(anchor_points[:n_p2][p_g_alt], target_box_alt, self.bbox_loss.dfl_loss.reg_max - 1)
                        pred_dist_alt = aux_logits[b, p_g_alt].view(-1, 4, self.bbox_loss.dfl_loss.reg_max)
                        loss_dfl = self.bbox_loss.dfl_loss(pred_dist_alt.flatten(0, 1), target_ltrb_alt)
                        loss_dfl = loss_dfl.squeeze(-1)
                        
                        gt_support_losses = loss_iou + loss_dfl
                        gt_loss = gt_support_losses.mean()
                        psd_loss = psd_loss + gt_loss
                        num_gt_losses += 1
                        
                        with torch.no_grad():
                            main_box_alt = pred_bboxes[b, :n_p2][p_g_alt]
                            iou_main = bbox_iou(main_box_alt, target_box_alt, xywh=False, CIoU=True)
                            total_iou_main += float(iou_main.sum().item())
                            total_iou_aux += float(iou_aux.sum().item())
                            total_iou_count += int(iou_aux.numel())
                            
                if num_gt_losses > 0:
                    psd_loss = psd_loss / num_gt_losses
                else:
                    psd_loss = psd_loss * 0.0
            
            delta_drop = (total_iou_main - total_iou_aux) / max(total_iou_count, 1)
            self.psd_metrics = {
                "psd_eligible_gt": total_eligible_gt,
                "psd_alt_supports_avg": total_alt_supports_found / max(total_eligible_gt, 1),
                "psd_raw_loss": float(psd_loss.detach().item()),
                "psd_delta_drop": delta_drop,
            }
        else:
            self.psd_metrics = {
                "psd_eligible_gt": 0,
                "psd_alt_supports_avg": 0.0,
                "psd_raw_loss": 0.0,
                "psd_delta_drop": 0.0,
            }

        psd_idx = (
            3
            + int(self.boundary_loss is not None)
            + int(self.loc_quality_loss is not None)
            + int(self.quality_head)
            + int(self.rank_gain > 0)
            + int(self.box_consensus_gain > 0)
        )
        if self.psd_enabled:
            current_psd_gain = self._psd_current_gain()
            loss[psd_idx] = psd_loss * current_psd_gain
            self.psd_metrics["psd_applied_loss"] = float(loss[psd_idx].detach().item())
            self.psd_metrics["psd_gain"] = current_psd_gain

        dgfe_idx = (
            3
            + int(self.boundary_loss is not None)
            + int(self.loc_quality_loss is not None)
            + int(self.quality_head)
            + int(self.rank_gain > 0)
            + int(self.box_consensus_gain > 0)
            + int(self.psd_enabled)
        )
        if self.dgfe_rec_gain > 0:
            loss[dgfe_idx] = self._dgfe_reconstruction_loss(preds, batch) * self.dgfe_rec_gain
            dgfe_idx += 1
        if self.dgfe_spatial_gain > 0:
            if fg_mask.sum() and assigned_iou is None:
                with torch.no_grad():
                    assigned_iou = bbox_iou(
                        pred_bboxes.detach()[fg_mask],
                        target_bboxes_scaled[fg_mask],
                        xywh=False,
                        CIoU=False,
                    ).squeeze(-1).clamp(0)
            q_by_gt = self._dgfe_quality_by_gt(
                assigned_iou, pred_bboxes, target_bboxes_scaled, target_gt_idx, fg_mask, mask_gt
            )
            loss[dgfe_idx] = self._dgfe_spatial_loss(preds, gt_bboxes, mask_gt, imgsz, q_by_gt) * self._dgfe_spatial_gain()
            dgfe_idx += 1
        if self.p2_detail_rec_gain > 0:
            loss[dgfe_idx] = self._p2_detail_reconstruction_loss(preds) * self.p2_detail_rec_gain
            self.p2_detail_metrics["p2_detail_applied_loss"] = float(loss[dgfe_idx].detach().item())
        self.localization_distill_metrics = {}
        if self.localization_distill_enabled:
            teacher = self.localization_distill_teacher
            base_dfl_loss = loss[2].detach()
            with torch.no_grad():
                teacher_output = teacher(batch["img"])
                teacher_preds = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
                teacher_logits = teacher_preds["boxes"].permute(0, 2, 1).contiguous()
            if teacher_logits.shape != pred_distri.shape:
                raise ValueError(
                    "Localization Distillation requires teacher/student DFL logits with identical shape; "
                    f"got {tuple(teacher_logits.shape)} vs {tuple(pred_distri.shape)}"
                )
            weighted_ld_loss = localization_distillation_loss(
                pred_distri, teacher_logits, fg_mask, self.localization_distill_lambda
            )
            ld_loss = weighted_ld_loss / max(self.localization_distill_lambda, 1e-12)
            loss[2] = loss[2] + weighted_ld_loss
            self.localization_distill_metrics = {
                "ld_loss": float(ld_loss.detach().item()),
                "ld_base_dfl_loss": float(base_dfl_loss.item()),
                "ld_teacher_student_divergence": float(ld_loss.detach().item()),
                "ld_active_positive_count": float(fg_mask.sum().item()),
                "ld_lambda": self.localization_distill_lambda,
            }
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl[, boundary][, loc_quality][, quality][, rank][, dgfe_rec][, dgfe_spatial])

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        
        # Calculate tiny-GT restraint loss for P1DRR modules
        restraint_loss = torch.tensor(0.0, device=self.device)
        drr_modules = [m for m in self.model.modules() if m.__class__.__name__ == "P1DRR"]
        
        if len(drr_modules) > 0 and "bboxes" in batch:
            H_img, W_img = batch["img"].shape[2:]
            
            for m in drr_modules:
                if m.gate_tensor is not None:
                    G = m.gate_tensor  # (B, 1, H, W)
                    B, _, H, W = G.shape
                    
                    # Create binary target mask for tiny objects (area < 400 px^2)
                    M_gt = torch.zeros((B, 1, H, W), device=self.device, dtype=torch.float32)
                    
                    if len(batch["bboxes"]) > 0:
                        for b in range(B):
                            img_boxes = batch["bboxes"][batch["batch_idx"] == b]
                            for box in img_boxes:
                                xc, yc, w, h = box
                                w_px = w * W_img
                                h_px = h * H_img
                                area = w_px * h_px
                                
                                if 0.0 < area < 400.0:
                                    # Convert xywh to xyxy
                                    x1 = xc - w / 2.0
                                    y1 = yc - h / 2.0
                                    x2 = xc + w / 2.0
                                    y2 = yc + h / 2.0
                                    
                                    # Map to grid coordinates
                                    gx1 = int(torch.clamp(x1 * W, min=0.0, max=W - 1).item())
                                    gy1 = int(torch.clamp(y1 * H, min=0.0, max=H - 1).item())
                                    gx2 = int(torch.clamp(x2 * W, min=0.0, max=W - 1).item())
                                    gy2 = int(torch.clamp(y2 * H, min=0.0, max=H - 1).item())
                                    
                                    # Dilate by 1 cell
                                    gx1_ex = max(0, gx1 - 1)
                                    gy1_ex = max(0, gy1 - 1)
                                    gx2_ex = min(W - 1, gx2 + 1)
                                    gy2_ex = min(H - 1, gy2 + 1)
                                    
                                    M_gt[b, 0, gy1_ex:gy2_ex+1, gx1_ex:gx2_ex+1] = 1.0
                                    
                    outside_mask = 1.0 - M_gt
                    loss_val = torch.sum(outside_mask * G) / (torch.sum(outside_mask) + 1e-6)
                    restraint_loss = restraint_loss + loss_val
            
            # Determine current training epoch for restraint warm-up
            epoch = 0
            if hasattr(self.model, "trainer") and self.model.trainer is not None:
                epoch = getattr(self.model.trainer, "epoch", 0)
            
            # lambda_r warmup: 0 for epoch < 3, then ramp up to 0.01 at epoch 6
            if epoch < 3:
                lambda_r = 0.0
            elif epoch < 6:
                lambda_r = 0.01 * (epoch - 3) / 3.0
            else:
                lambda_r = 0.01
                
            if restraint_loss > 0 and lambda_r > 0:
                loss[1] = loss[1] + lambda_r * restraint_loss
                loss_detach = loss_detach.clone()
                loss_detach[1] = loss_detach[1] + lambda_r * restraint_loss.item()
                
        return loss * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semantic
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semantic = proto
        else:
            pred_semantic = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semantic is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semantic)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLO26 pose estimation with RLE loss support."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        if not kpt_mask.any():
            return pred_kpt[..., :0].sum()

        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        if not error.numel():
            return pred_kpt[..., :0].sum()

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return pred_kpt[..., :0].sum()

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=float(getattr(self.hyp, "tal_alpha", 0.5)),
            beta=float(getattr(self.hyp, "tal_beta", 6.0)),
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xywhr, (b, h*w, 5)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        self.dbss_assignment_context = self.one2many.dbss_assignment_context
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model: torch.nn.Module, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]


class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with ignore_index=255.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255).to(device=self.device, dtype=self.dtype)
            if self.use_cityscapes_weight:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                weight = torch.from_numpy(CITYSCAPES_WEIGHT).to(device=self.device, dtype=self.dtype)
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """Resize masks to match prediction spatial dimensions."""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        if self.nc == 1:
            flat = masks.reshape(-1)
            valid = flat != 255
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = masks.reshape(-1).long()
        return self.ce(logits, target)

    def _dice_loss(self, preds, masks):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks)
        flat_target = masks.reshape(-1)
        valid = flat_target != 255
        if not valid.any():
            return preds.sum() * 0

        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks):
        """Compute Dice loss for single-class (binary) segmentation.

        Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.
        """
        valid = (masks != 255).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss.

        Args:
            preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
            batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): (total_loss * batch_size, detached loss items [ce, dice, aux]).
        """
        # Unpack auxiliary logits when present.
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # Main cross-entropy and Dice loss.
        ce_loss = self._ce_loss(preds, masks)
        dice_loss = self._dice_loss(preds, masks)
        total = ce_loss + dice_loss

        # Auxiliary cross-entropy loss. Match ce_loss dtype so torch.stack below succeeds under AMP.
        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, masks) * 0.4
            total += aux_loss

        loss_items = torch.stack([ce_loss, dice_loss, aux_loss]).detach()
        return total * preds.shape[0], loss_items
