# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FTSC modules for tiny-object FPN supervision experiments.

The modules are independent from the older DALA/DBSS experiments. The
assignment-preserving calibrator is owned by Detect but runs only in the loss
after TAL, while the legacy detail calibrator remains a plug-in feature layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "AnchorFreeFTSCCalibrator",
    "DCFLDGMMQualityEvidence",
    "DEIMPositiveMALLoss",
    "DFLDistributionEvidence",
    "DFLUncertaintyMinimization",
    "FCOSCenternessEvidence",
    "FTSCFeatureCalibrator",
    "HierarchicalBackgroundSmoothing",
    "PositionGaussianEvidence",
)


class DEIMPositiveMALLoss(nn.Module):
    """DEIM-inspired positive-only Matchability-Aware Loss adaptation.

    Full DEIM also changes dense negative supervision and matching. This
    adaptation deliberately does neither: it replaces only TAL positive
    target-class BCE elements with BCE targets ``q**gamma``.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        if gamma <= 0:
            raise ValueError("DEIM-MAL gamma must be positive.")
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """Return numerically stable per-element MAL with detached IoU quality."""
        quality_target = quality.detach().to(device=logits.device, dtype=logits.dtype).clamp(0, 1).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(logits, quality_target, reduction="none")

    def replace_dense(
        self,
        base_loss: torch.Tensor,
        logits: torch.Tensor,
        positive_target_mask: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace only positive target-class elements and preserve every other BCE element exactly."""
        if int(positive_target_mask.sum().item()) != quality.numel():
            raise ValueError("DEIM-MAL requires one detached IoU quality per positive target-class element.")
        positive_loss = self(logits[positive_target_mask], quality)
        replaced = base_loss.clone()
        replaced[positive_target_mask] = positive_loss
        return replaced, positive_loss


class DFLUncertaintyMinimization(nn.Module):
    """UGS-inspired direct entropy minimization with an optional FTSC quality guard.

    This implements only uncertainty minimization. UGS feature perturbation and
    uncertainty-guided refinement are intentionally omitted. The quality guard
    is an FTSC adaptation and is not part of the reproduced UGS mechanism.
    """

    def __init__(
        self,
        reg_max: int,
        normalized: bool = True,
        eps: float = 1e-9,
        quality_guard: bool = False,
        quality_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if reg_max <= 1:
            raise ValueError("DFL uncertainty minimization requires reg_max > 1.")
        if quality_gamma <= 0:
            raise ValueError("DFL uncertainty-minimization quality gamma must be positive.")
        self.reg_max = int(reg_max)
        self.normalized = bool(normalized)
        self.quality_guard = bool(quality_guard)
        self.quality_gamma = float(quality_gamma)
        self.eps = float(eps)
        self.last_raw_entropy: torch.Tensor | None = None
        self.last_normalized_entropy: torch.Tensor | None = None
        self.last_positive_entropy: torch.Tensor | None = None
        self.last_quality: torch.Tensor | None = None
        self.last_guard: torch.Tensor | None = None

    def forward(
        self,
        pred_distri: torch.Tensor,
        fg_mask: torch.Tensor,
        localization_quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return positive DFL entropy while retaining gradient only through the entropy path."""
        logits = pred_distri[fg_mask].reshape(-1, 4, self.reg_max).float()
        if not logits.numel():
            self.last_raw_entropy = logits.new_empty(0)
            self.last_normalized_entropy = logits.new_empty(0)
            self.last_positive_entropy = logits.new_empty(0)
            self.last_quality = logits.new_empty(0)
            self.last_guard = logits.new_empty(0)
            return pred_distri.sum() * 0.0
        probabilities = logits.softmax(-1)
        raw_entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(-1)
        normalized_entropy = raw_entropy / math.log(self.reg_max)
        self.last_raw_entropy = raw_entropy.detach()
        self.last_normalized_entropy = normalized_entropy.detach()
        entropy = normalized_entropy if self.normalized else raw_entropy
        positive_entropy = entropy.mean(-1)
        self.last_positive_entropy = positive_entropy.detach()
        if not self.quality_guard:
            self.last_quality = None
            self.last_guard = None
            # Preserve the exact R6 reduction when the new option is disabled.
            return entropy.mean()
        if localization_quality is None:
            raise ValueError("Quality-guarded DFL uncertainty minimization requires realized IoU.")
        if localization_quality.numel() != positive_entropy.numel():
            raise ValueError("Quality-guarded DFL uncertainty minimization requires one IoU per positive.")
        quality = localization_quality.detach().to(device=logits.device, dtype=positive_entropy.dtype).clamp(0, 1)
        guard = quality.pow(self.quality_gamma)
        self.last_quality = quality
        self.last_guard = guard
        return (guard * positive_entropy).sum() / guard.sum().clamp_min(self.eps)


class HierarchicalBackgroundSmoothing(nn.Module):
    """SET-style adaptive smoothing for the GT-masked background of one FPN level.

    This module implements Eqs. (1)--(4) of Sun et al., *SET: Spectral
    Enhancement for Tiny Object Detection* (CVPR 2025). It is owned by the
    Detect head but is called only by the auxiliary training path.
    """

    def __init__(
        self,
        channels: int,
        stride: int,
        reduction: int = 4,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("HBS channels must be positive.")
        if stride <= 0:
            raise ValueError("HBS stride must be positive.")
        if reduction <= 0:
            raise ValueError("HBS channel reduction must be positive.")
        if kernel_size is None:
            kernel_size = math.ceil(math.log2(stride) / 2) * 2 + 1
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("HBS kernel_size must be a positive odd integer.")

        hidden = max(channels // reduction, 1)
        padding = kernel_size // 2
        self.channels = int(channels)
        self.stride = int(stride)
        self.reduction = int(reduction)
        self.kernel_size = int(kernel_size)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size, padding=padding)
        self.expand = nn.Conv2d(hidden, channels, kernel_size, padding=padding)

    @staticmethod
    def foreground_mask(
        feature: torch.Tensor,
        batch_idx: torch.Tensor | None,
        bboxes: torch.Tensor | None,
    ) -> torch.Tensor:
        """Rasterize normalized xywh GT boxes as a union mask at feature resolution."""
        mask = feature.new_zeros((feature.shape[0], 1, feature.shape[2], feature.shape[3]))
        if batch_idx is None or bboxes is None or bboxes.numel() == 0:
            return mask

        indices = batch_idx.detach().to(device=feature.device, dtype=torch.long).view(-1)
        boxes = bboxes.detach().to(device=feature.device, dtype=torch.float32).view(-1, 4)
        if indices.numel() != boxes.shape[0]:
            raise ValueError("HBS batch_idx and bboxes must contain the same number of labels.")

        height, width = feature.shape[-2:]
        for image_index, box in zip(indices.tolist(), boxes, strict=False):
            if image_index < 0 or image_index >= feature.shape[0]:
                raise ValueError(f"HBS batch index {image_index} is outside batch size {feature.shape[0]}.")
            xc, yc, box_width, box_height = box.tolist()
            left = max(0.0, min(1.0, xc - box_width / 2))
            top = max(0.0, min(1.0, yc - box_height / 2))
            right = max(0.0, min(1.0, xc + box_width / 2))
            bottom = max(0.0, min(1.0, yc + box_height / 2))
            if right <= left or bottom <= top:
                continue
            x1 = max(0, min(width - 1, math.floor(left * width)))
            y1 = max(0, min(height - 1, math.floor(top * height)))
            x2 = max(x1 + 1, min(width, math.ceil(right * width)))
            y2 = max(y1 + 1, min(height, math.ceil(bottom * height)))
            mask[image_index, :, y1:y2, x1:x2] = 1
        return mask

    def forward(
        self,
        feature: torch.Tensor,
        batch_idx: torch.Tensor | None,
        bboxes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Eq. (2) HBS-enhanced features and their foreground mask."""
        mask = self.foreground_mask(feature, batch_idx, bboxes)
        foreground = feature * mask
        background = feature * (1 - mask)
        smoothed_background = F.relu(self.expand(F.relu(self.reduce(background)))) + background
        return foreground + smoothed_background, mask


class PositionGaussianEvidence(nn.Module):
    """Return stable log-evidence from an anchor point's position inside its assigned GT box."""

    def __init__(self, alpha: float = 6.0, eps: float = 1e-9) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("FTSC position alpha must be positive.")
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        anchor_points_px: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log Position-Gaussian evidence for TAL positives only."""
        points = anchor_points_px.unsqueeze(0).expand(target_bboxes_px.shape[0], -1, -1)[fg_mask]
        boxes = target_bboxes_px[fg_mask]
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(self.eps)
        normalized_offset = (points - centers) / (sizes / self.alpha).clamp_min(self.eps)
        return -0.5 * normalized_offset.square().sum(-1)


class FCOSCenternessEvidence(nn.Module):
    """Return log FCOS centerness for TAL positives without adding an FCOS head.

    This is a post-assignment FTSC adaptation of FCOS Eq. (3), not an FCOS
    reproduction: the target-derived score is centered per GT and used only by
    the training-time F5 calibrator.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        anchor_points_px: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        points = anchor_points_px.unsqueeze(0).expand(target_bboxes_px.shape[0], -1, -1)[fg_mask]
        boxes = target_bboxes_px[fg_mask]
        left_top = (points - boxes[:, :2]).clamp_min(0.0)
        right_bottom = (boxes[:, 2:] - points).clamp_min(0.0)
        left, top = left_top.unbind(-1)
        right, bottom = right_bottom.unbind(-1)
        horizontal = torch.minimum(left, right) / torch.maximum(left, right).clamp_min(self.eps)
        vertical = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp_min(self.eps)
        return 0.5 * (horizontal.clamp_min(self.eps).log() + vertical.clamp_min(self.eps).log())


class DFLDistributionEvidence(nn.Module):
    """Return detached log-reliability from entropy/variance of positive DFL distributions."""

    def __init__(
        self,
        reg_max: int,
        entropy_tau: float = 1.0,
        variance_tau: float = 0.0,
        detach: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if reg_max <= 1:
            raise ValueError("DFL evidence requires reg_max > 1.")
        if entropy_tau < 0 or variance_tau < 0:
            raise ValueError("DFL entropy/variance strengths must be non-negative.")
        self.reg_max = int(reg_max)
        self.entropy_tau = float(entropy_tau)
        self.variance_tau = float(variance_tau)
        self.detach = bool(detach)
        self.eps = float(eps)
        self.register_buffer("bins", torch.arange(self.reg_max, dtype=torch.float32), persistent=False)
        self.last_entropy: torch.Tensor | None = None
        self.last_variance: torch.Tensor | None = None

    def forward(self, pred_distri: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """Compute one log-reliability value per TAL positive."""
        logits = pred_distri[fg_mask].reshape(-1, 4, self.reg_max).float()
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(-1) / math.log(self.reg_max)
        bins = self.bins.to(device=probabilities.device)
        expected = (probabilities * bins).sum(-1, keepdim=True)
        variance = (probabilities * (bins - expected).square()).sum(-1)
        variance = variance / max(float((self.reg_max - 1) ** 2), self.eps)
        entropy = entropy.mean(-1)
        variance = variance.mean(-1)
        log_evidence = -self.entropy_tau * entropy - self.variance_tau * variance
        if self.detach:
            log_evidence = log_evidence.detach()
        self.last_entropy = entropy.detach()
        self.last_variance = variance.detach()
        return log_evidence.to(dtype=pred_distri.dtype)


class DCFLDGMMQualityEvidence(nn.Module):
    """DCFL-inspired per-GT DGMM relative-quality evidence provider.

    DCFL's candidate expansion and coarse assignment are intentionally omitted.
    For the already-selected TAL positives, this parameter-free provider builds
    ``PT = 0.5 * (assigned-class probability + realized IoU)`` from detached
    predictions. It estimates two modes with a single vectorized moment fit,
    then returns a bounded high-vs-low log likelihood. Canonical FTSC performs
    the subsequent per-GT centering and F5 gating.
    """

    def __init__(
        self,
        min_group_size: int = 4,
        min_variance: float = 1e-4,
        evidence_clip: float = 3.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if min_group_size < 2:
            raise ValueError("DCFL-DGMM min_group_size must be at least two.")
        if min_variance <= 0 or evidence_clip <= 0:
            raise ValueError("DCFL-DGMM variance and evidence clip must be positive.")
        self.min_group_size = int(min_group_size)
        self.min_variance = float(min_variance)
        self.evidence_clip = float(evidence_clip)
        self.eps = float(eps)
        self.last_pt: torch.Tensor | None = None
        self.last_raw_evidence: torch.Tensor | None = None
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
        values = values.detach().float()
        if not values.numel():
            return {f"{prefix}_{suffix}": 0.0 for suffix in ("mean", "std", "min", "max")}
        return {
            f"{prefix}_mean": float(values.mean().item()),
            f"{prefix}_std": float(values.std(unbiased=False).item()),
            f"{prefix}_min": float(values.min().item()),
            f"{prefix}_max": float(values.max().item()),
        }

    def forward(
        self,
        classification_quality: torch.Tensor,
        localization_quality: torch.Tensor,
        group_ids: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        """Return detached bounded raw quality evidence for canonical FTSC centering."""
        if classification_quality.numel() != localization_quality.numel():
            raise ValueError("DCFL-DGMM quality inputs and group IDs must have matching lengths.")
        if classification_quality.numel() != group_ids.numel():
            raise ValueError("DCFL-DGMM requires one group ID per positive quality value.")
        with torch.no_grad():
            pt = 0.5 * (
                classification_quality.detach().float().clamp(0, 1)
                + localization_quality.detach().float().clamp(0, 1)
            )
            if not pt.numel():
                self.last_pt = pt
                self.last_raw_evidence = pt
                self.last_metrics = {
                    "ftsc_dcfl_dgmm_fit_fraction": 0.0,
                    "ftsc_dcfl_fallback_fraction": 0.0,
                }
                return pt.to(dtype=classification_quality.dtype)

            ones = torch.ones_like(pt)
            counts = pt.new_zeros(group_count).scatter_add_(0, group_ids, ones)
            sums = pt.new_zeros(group_count).scatter_add_(0, group_ids, pt)
            square_sums = pt.new_zeros(group_count).scatter_add_(0, group_ids, pt.square())
            means = sums / counts.clamp_min(1.0)
            variances = (square_sums / counts.clamp_min(1.0) - means.square()).clamp_min(0.0)
            active = counts > 0
            eligible = (counts >= self.min_group_size) & (variances >= self.min_variance)

            high_mask = pt > means[group_ids]
            high = high_mask.to(pt.dtype)
            low = 1.0 - high
            high_counts = pt.new_zeros(group_count).scatter_add_(0, group_ids, high)
            low_counts = counts - high_counts
            fitted = eligible & (high_counts > 0) & (low_counts > 0)

            high_sums = pt.new_zeros(group_count).scatter_add_(0, group_ids, pt * high)
            low_sums = sums - high_sums
            high_means = high_sums / high_counts.clamp_min(1.0)
            low_means = low_sums / low_counts.clamp_min(1.0)
            high_square_sums = pt.new_zeros(group_count).scatter_add_(0, group_ids, pt.square() * high)
            low_square_sums = square_sums - high_square_sums
            high_variances = (high_square_sums / high_counts.clamp_min(1.0) - high_means.square()).clamp_min(0.0)
            low_variances = (low_square_sums / low_counts.clamp_min(1.0) - low_means.square()).clamp_min(0.0)
            high_variances = torch.where(high_counts > 1, high_variances, variances).clamp_min(self.min_variance)
            low_variances = torch.where(low_counts > 1, low_variances, variances).clamp_min(self.min_variance)

            group = group_ids
            log_high = (
                -0.5 * ((pt - high_means[group]).square() / high_variances[group] + high_variances[group].log())
                + (high_counts[group] / counts[group].clamp_min(1.0)).clamp_min(self.eps).log()
            )
            log_low = (
                -0.5 * ((pt - low_means[group]).square() / low_variances[group] + low_variances[group].log())
                + (low_counts[group] / counts[group].clamp_min(1.0)).clamp_min(self.eps).log()
            )
            dgmm_evidence = log_high - log_low
            fallback = (pt - means[group]) / variances[group].clamp_min(self.min_variance).sqrt()
            evidence = torch.where(fitted[group], dgmm_evidence, fallback)
            evidence = torch.where(counts[group] > 1, evidence, torch.zeros_like(evidence))
            evidence = torch.nan_to_num(evidence).clamp(-self.evidence_clip, self.evidence_clip)

            active_count = int(active.sum().item())
            fit_count = int(fitted.sum().item())
            self.last_pt = pt
            self.last_raw_evidence = evidence
            self.last_metrics = {
                "ftsc_dcfl_dgmm_fit_fraction": fit_count / max(active_count, 1),
                "ftsc_dcfl_fallback_fraction": (active_count - fit_count) / max(active_count, 1),
            }
            self.last_metrics.update(self._stats("ftsc_dcfl_pt", pt))
            self.last_metrics.update(self._stats("ftsc_dcfl_provider_raw", evidence))
        return evidence.to(device=classification_quality.device, dtype=classification_quality.dtype)


class AnchorFreeFTSCCalibrator(nn.Module):
    """Assignment-preserving supervision calibration for anchor-free YOLO detection.

    The module is owned by the Detect head so F5 evidence-strength parameters exist
    before the trainer builds its optimizer. It is called only by the criterion
    after TAL assignment and therefore adds no inference-time work.
    """

    SUPPORTED_EVIDENCE = {"position_gaussian", "fcos_centerness", "dcfl_dgmm_quality", "dfl_distribution"}
    # Providers in this set use canonical Position-provider routing: cls + box
    # + DFL, with the same F5 strength, centering, clipping and scheduling.
    POSITION_EVIDENCE = {"position_gaussian", "fcos_centerness", "dcfl_dgmm_quality"}
    TASK_NAMES = ("cls", "box", "dfl")

    def __init__(self, config: dict, reg_max: int) -> None:
        super().__init__()
        config = dict(config or {})
        self.policy = str(config.get("policy", "e4")).lower()
        if self.policy not in {"e4", "f5"}:
            raise ValueError("FTSC policy must be 'e4' or 'f5'.")

        evidence = config.get("evidence", ["position_gaussian"])
        self.evidence_names = tuple(str(name).lower() for name in evidence)
        unknown = set(self.evidence_names) - self.SUPPORTED_EVIDENCE
        if unknown:
            raise ValueError(f"Unsupported FTSC evidence: {sorted(unknown)}")
        if not self.evidence_names:
            raise ValueError("FTSC requires at least one evidence provider.")
        if len(set(self.evidence_names) & self.POSITION_EVIDENCE) > 1:
            raise ValueError("FTSC accepts exactly one Position/centrality evidence provider per experiment.")

        self.log_clip = float(config.get("log_clip", 0.35))
        if self.log_clip <= 0:
            raise ValueError("FTSC log_clip must be positive.")
        self.per_gt_norm = bool(config.get("per_gt_norm", True))
        self.warmup_epochs = max(int(config.get("warmup_epochs", 5)), 0)
        self.ramp_epochs = max(int(config.get("ramp_epochs", 10)), 1)
        self.apply_cls = bool(config.get("apply_cls", True))
        self.apply_box = bool(config.get("apply_box", True))
        self.apply_dfl = bool(config.get("apply_dfl", True))
        self.dfl_apply_cls = bool(config.get("dfl_apply_cls", True))
        self.dfl_apply_box = bool(config.get("dfl_apply_box", False))
        self.dfl_apply_dfl = bool(config.get("dfl_apply_dfl", False))
        self.position_task_specific_strength = bool(config.get("position_task_specific_strength", False))
        self.classification_replacement = str(config.get("classification_replacement", "bce")).lower()
        if self.classification_replacement not in {"bce", "deim_mal"}:
            raise ValueError("FTSC classification_replacement must be 'bce' or 'deim_mal'.")
        self.mal_gamma = float(config.get("mal_gamma", 1.0))
        self.mal_loss = (
            DEIMPositiveMALLoss(self.mal_gamma) if self.classification_replacement == "deim_mal" else None
        )
        self.um_enabled = bool(config.get("uncertainty_minimization", False))
        self.um_lambda = float(config.get("um_lambda", 0.05))
        self.um_normalized = bool(config.get("um_normalized", True))
        self.um_quality_guard = bool(config.get("um_quality_guard", False))
        self.um_quality_gamma = float(config.get("um_quality_gamma", 1.0))
        if self.um_lambda < 0:
            raise ValueError("UGS-UM lambda must be non-negative.")
        if self.um_quality_gamma <= 0:
            raise ValueError("UGS-UM quality gamma must be positive.")
        if self.um_quality_guard and not self.um_enabled:
            raise ValueError("UGS-UM quality guard requires uncertainty_minimization=true.")
        if self.classification_replacement == "deim_mal" and "dfl_distribution" in self.evidence_names:
            raise ValueError("DEIM-MAL replaces the DFL-to-classification evidence provider; remove it from evidence.")
        if self.um_enabled and "dfl_distribution" in self.evidence_names:
            raise ValueError("UGS-UM replaces the DFL-to-classification evidence provider; remove it from evidence.")
        if self.um_enabled and self.classification_replacement != "bce":
            raise ValueError("Paper-replacement ablations must not combine UGS-UM with DEIM-MAL.")
        self.dfl_shuffle_within_gt = bool(config.get("dfl_shuffle_within_gt", False))
        self.gt_mass_rebalance_cls = bool(config.get("gt_mass_rebalance_cls", False))
        self.gt_mass_power = float(config.get("gt_mass_power", 0.5))
        self.gt_mass_min_factor = float(config.get("gt_mass_min_factor", 0.75))
        self.gt_mass_max_factor = float(config.get("gt_mass_max_factor", 1.25))
        self.gt_mass_shuffle = bool(config.get("gt_mass_shuffle", False))
        self.strength_max = float(config.get("strength_max", 2.0))
        self.strength_init = float(config.get("strength_init", 1.0))
        self.strength_reg_weight = float(config.get("strength_reg_weight", 1e-4))
        configured_fixed_strengths = config.get("fixed_strengths", {})
        if not isinstance(configured_fixed_strengths, dict):
            raise ValueError("FTSC fixed_strengths must be a mapping of evidence name to scalar value.")
        self.fixed_strengths = {str(name).lower(): float(value) for name, value in configured_fixed_strengths.items()}
        if not 0 < self.strength_init < self.strength_max:
            raise ValueError("FTSC strength_init must lie strictly between zero and strength_max.")
        if self.strength_reg_weight < 0:
            raise ValueError("FTSC strength_reg_weight must be non-negative.")
        unknown_fixed_strengths = set(self.fixed_strengths) - set(self.evidence_names)
        if unknown_fixed_strengths:
            raise ValueError(
                f"Fixed FTSC strengths require active evidence providers: {sorted(unknown_fixed_strengths)}"
            )
        if self.fixed_strengths and self.policy != "f5":
            raise ValueError("Explicit fixed FTSC strengths are only supported by the F5 policy.")
        if any(not 0 < value <= self.strength_max for value in self.fixed_strengths.values()):
            raise ValueError(f"Fixed FTSC strengths must satisfy 0 < value <= strength_max ({self.strength_max}).")
        if self.position_task_specific_strength and not (set(self.evidence_names) & self.POSITION_EVIDENCE):
            raise ValueError("Task-specific Position strength requires a Position/centrality evidence provider.")
        if self.position_task_specific_strength and self.policy != "f5":
            raise ValueError("Task-specific Position strength requires the learnable F5 policy.")
        if self.dfl_shuffle_within_gt and "dfl_distribution" not in self.evidence_names:
            raise ValueError("Within-GT DFL shuffle requires dfl_distribution evidence.")
        if self.dfl_shuffle_within_gt and not bool(config.get("dfl_detach", True)):
            raise ValueError("Within-GT DFL shuffle requires detached DFL evidence.")
        if self.gt_mass_power < 0:
            raise ValueError("FTSC GT-mass power must be non-negative.")
        if not 0 < self.gt_mass_min_factor <= 1.0 <= self.gt_mass_max_factor:
            raise ValueError("FTSC GT-mass factors must satisfy 0 < min <= 1 <= max.")
        if self.gt_mass_shuffle and not self.gt_mass_rebalance_cls:
            raise ValueError("FTSC GT-mass shuffle requires gt_mass_rebalance_cls=true.")

        self.position_tasks = (
            tuple(
                task
                for task, enabled in zip(self.TASK_NAMES, (self.apply_cls, self.apply_box, self.apply_dfl))
                if enabled
            )
            if set(self.evidence_names) & self.POSITION_EVIDENCE
            else ()
        )
        if self.position_task_specific_strength and not self.position_tasks:
            raise ValueError("Task-specific Position strength requires at least one enabled Position task.")

        if self.dfl_shuffle_within_gt:
            configured_seed = config.get("dfl_shuffle_seed")
            shuffle_seed = (torch.initial_seed() if configured_seed is None else int(configured_seed)) % (2**63 - 1)
            self.register_buffer("dfl_shuffle_seed", torch.tensor(shuffle_seed, dtype=torch.long))
            self.register_buffer("dfl_shuffle_step", torch.tensor(0, dtype=torch.long))
        self._last_dfl_shuffle_metrics: dict[str, float] = {}
        # Optional post-hoc/train-time causal audit.  It is disabled by
        # default and never participates in the loss graph or inference.
        self.audit_enabled = bool(config.get("audit_enabled", False))
        self.last_audit: dict[str, torch.Tensor] = {}
        if self.gt_mass_shuffle:
            configured_seed = config.get("gt_mass_shuffle_seed")
            shuffle_seed = (torch.initial_seed() if configured_seed is None else int(configured_seed)) % (2**63 - 1)
            self.register_buffer("gt_mass_shuffle_seed", torch.tensor(shuffle_seed, dtype=torch.long))
            self.register_buffer("gt_mass_shuffle_step", torch.tensor(0, dtype=torch.long))
        self._last_gt_mass_metrics: dict[str, float] = {}

        providers = {}
        if "position_gaussian" in self.evidence_names:
            providers["position_gaussian"] = PositionGaussianEvidence(float(config.get("position_alpha", 6.0)))
        if "fcos_centerness" in self.evidence_names:
            providers["fcos_centerness"] = FCOSCenternessEvidence()
        if "dcfl_dgmm_quality" in self.evidence_names:
            providers["dcfl_dgmm_quality"] = DCFLDGMMQualityEvidence(
                min_group_size=int(config.get("dcfl_min_group_size", 4)),
                min_variance=float(config.get("dcfl_min_variance", 1e-4)),
                evidence_clip=float(config.get("dcfl_evidence_clip", 3.0)),
            )
        if "dfl_distribution" in self.evidence_names:
            providers["dfl_distribution"] = DFLDistributionEvidence(
                reg_max=reg_max,
                entropy_tau=float(config.get("dfl_entropy_tau", 1.0)),
                variance_tau=float(config.get("dfl_variance_tau", 0.0)),
                detach=bool(config.get("dfl_detach", True)),
            )
        self.providers = nn.ModuleDict(providers)
        self.um_regularizer = (
            DFLUncertaintyMinimization(
                reg_max,
                normalized=self.um_normalized,
                quality_guard=self.um_quality_guard,
                quality_gamma=self.um_quality_gamma,
            )
            if self.um_enabled
            else None
        )

        self.strength_logits = nn.ParameterDict()
        if self.policy == "f5":
            probability = self.strength_init / self.strength_max
            initial_logit = math.log(probability / (1.0 - probability))
            for name in self.evidence_names:
                if name in self.fixed_strengths:
                    continue
                for key in self.strength_keys(name):
                    self.strength_logits[key] = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _audit_cpu(value: torch.Tensor) -> torch.Tensor:
        """Detach an audit tensor before it leaves the loss call."""
        return value.detach().float().cpu()

    def _make_audit_snapshot(
        self,
        *,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_scores: torch.Tensor,
        stride: torch.Tensor,
        anchor_index: torch.Tensor,
        audit_decoded_iou: torch.Tensor | None,
        raw_evidence: dict[str, torch.Tensor],
        task_logs: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
        dfl_entropy: torch.Tensor | None,
        dfl_variance: torch.Tensor | None,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """Materialize per-positive audit columns with no autograd edges."""
        with torch.no_grad():
            positive = fg_mask
            batch_ids = torch.arange(fg_mask.shape[0], device=fg_mask.device).view(-1, 1).expand_as(fg_mask)
            image_index = batch_ids[positive].long()
            gt_id = target_gt_idx[positive].long()
            group_ids, group_count = self._positive_group_ids(fg_mask, target_gt_idx)
            counts = torch.zeros(group_count, device=fg_mask.device, dtype=torch.long)
            counts.scatter_add_(0, group_ids, torch.ones_like(group_ids, dtype=torch.long))
            gt_positive_count = counts[group_ids]
            stride_dense = stride.view(1, -1).expand_as(fg_mask) if stride.ndim == 1 else stride
            stride_positive = stride_dense[positive].detach().to(dtype=torch.float32)
            output: dict[str, torch.Tensor] = {
                "image_index": image_index,
                "gt_id": gt_id,
                "positive_index": anchor_index[positive].long(),
                "stride": stride_positive,
                "gt_positive_count": gt_positive_count,
                "tal_target_score": target_scores.sum(-1)[positive],
                "tal_assigned_class_score": target_scores[positive].amax(-1),
                "tal_assigned_class": target_scores[positive].argmax(-1).long(),
                "audit_decoded_iou": (
                    audit_decoded_iou
                    if audit_decoded_iou is not None
                    else target_scores.new_full((int(positive.sum().item()),), float("nan"))
                ),
                "epoch": torch.full(
                    (int(positive.sum().item()),), int(epoch), device=fg_mask.device, dtype=torch.int64
                ),
                "ftsc_log_clip": target_scores.new_full(
                    (int(positive.sum().item()),), float(self.log_clip)
                ),
                "ftsc_rho": target_scores.new_full(
                    (int(positive.sum().item()),), float(self.residual_fraction(epoch))
                ),
                "dfl_variance_tau": target_scores.new_full(
                    (int(positive.sum().item()),),
                    float(
                        getattr(
                            self.providers["dfl_distribution"] if "dfl_distribution" in self.providers else None,
                            "variance_tau",
                            0.0,
                        )
                    ),
                ),
            }
            for name, values in raw_evidence.items():
                key = "position" if name in self.POSITION_EVIDENCE else "dfl"
                output[f"{key}_log_evidence_raw"] = values
                # Keep the audit definition centered even when an ablation
                # disables FTSC's per-GT normalization for its actual loss.
                output[f"{key}_log_evidence_centered"] = self._center_per_gt(
                    values, fg_mask, target_gt_idx
                )
            if dfl_entropy is not None:
                output["dfl_entropy_mean"] = dfl_entropy
            if dfl_variance is not None:
                output["dfl_variance_mean"] = dfl_variance
            for task in self.TASK_NAMES:
                output[f"w_{task}"] = weights[task]
                output[f"log_w_{task}"] = task_logs[task]
                output[f"w_{task}_clipped_low"] = (task_logs[task] < -self.log_clip).to(torch.float32)
                output[f"w_{task}_clipped_high"] = (task_logs[task] > self.log_clip).to(torch.float32)
            for name in self.evidence_names:
                if name in self.POSITION_EVIDENCE and self.position_task_specific_strength:
                    strengths = [
                        self.strength(name, target_scores.new_ones(1), task=task).detach().reshape(1)
                        for task in self.position_tasks
                    ]
                    for task, strength in zip(self.position_tasks, strengths, strict=False):
                        output[f"ftsc_strength_{name}_{task}"] = strength.expand(int(positive.sum().item()))
                    output[f"ftsc_strength_{name}"] = torch.stack(strengths).mean().expand(
                        int(positive.sum().item())
                    )
                else:
                    strength = self.strength(name, target_scores.new_ones(1)).detach().reshape(1)
                    output[f"ftsc_strength_{name}"] = strength.expand(int(positive.sum().item()))
            return {name: self._audit_cpu(value) for name, value in output.items()}

    def strength_keys(self, name: str) -> tuple[str, ...]:
        """Return parameter keys owned by one evidence provider."""
        if name in self.POSITION_EVIDENCE and self.position_task_specific_strength:
            return tuple(f"{name}_{task}" for task in self.position_tasks)
        return (name,)

    def strength(self, name: str, reference: torch.Tensor, task: str | None = None) -> torch.Tensor:
        """Return fixed E4 strength or bounded learnable F5 strength."""
        if self.policy == "e4":
            return reference.new_tensor(self.strength_init)
        if name in self.fixed_strengths:
            if task is not None:
                raise ValueError(f"Fixed shared {name} strength does not accept task={task!r}.")
            return reference.new_tensor(self.fixed_strengths[name])
        if name in self.POSITION_EVIDENCE and self.position_task_specific_strength:
            if task not in self.position_tasks:
                raise ValueError(
                    f"Position strength requires one of the enabled tasks {self.position_tasks}, got {task!r}."
                )
            key = f"{name}_{task}"
        else:
            if task is not None:
                raise ValueError(f"Shared {name} strength does not accept task={task!r}.")
            key = name
        return torch.sigmoid(self.strength_logits[key]).to(reference) * self.strength_max

    def _strength_regularization(self, reference: torch.Tensor) -> torch.Tensor:
        """Return matched F5 regularization, averaging task-specific Position terms."""
        regularization = reference.sum() * 0.0
        if self.policy != "f5" or not self.strength_reg_weight:
            return regularization

        provider_terms = []
        for name in self.evidence_names:
            if name in self.fixed_strengths:
                continue
            if name in self.POSITION_EVIDENCE and self.position_task_specific_strength:
                task_terms = torch.stack(
                    [
                        (self.strength(name, reference, task=task) - self.strength_init).square()
                        for task in self.position_tasks
                    ]
                )
                provider_terms.append(task_terms.mean())
            else:
                provider_terms.append((self.strength(name, reference) - self.strength_init).square())
        return (
            self.strength_reg_weight * torch.stack(provider_terms).sum()
            if provider_terms
            else regularization
        )

    def residual_fraction(self, epoch: int) -> float:
        """Return the F5 identity-to-full-gate schedule for the current zero-based epoch."""
        if self.policy == "e4":
            return 1.0
        if epoch < self.warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)

    @staticmethod
    def _positive_group_ids(fg_mask: torch.Tensor, target_gt_idx: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return collision-free image/GT group IDs in positive flatten order."""
        batch_size, num_anchors = fg_mask.shape
        batch_ids = torch.arange(batch_size, device=fg_mask.device).view(-1, 1).expand_as(fg_mask)[fg_mask]
        return batch_ids * num_anchors + target_gt_idx[fg_mask].long(), batch_size * num_anchors

    @staticmethod
    def _center_per_gt(
        values: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Center log-evidence independently inside each image/GT positive group."""
        if not values.numel():
            return values
        group_ids, group_count = AnchorFreeFTSCCalibrator._positive_group_ids(fg_mask, target_gt_idx)
        sums = values.new_zeros(group_count).scatter_add_(0, group_ids, values)
        counts = values.new_zeros(group_count).scatter_add_(0, group_ids, torch.ones_like(values))
        return values - sums[group_ids] / counts[group_ids].clamp_min(1.0)

    @staticmethod
    def _normalize_per_gt(
        values: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the arithmetic mean weight of every image/GT group to one."""
        if not values.numel():
            return values
        group_ids, group_count = AnchorFreeFTSCCalibrator._positive_group_ids(fg_mask, target_gt_idx)
        sums = values.new_zeros(group_count).scatter_add_(0, group_ids, values)
        counts = values.new_zeros(group_count).scatter_add_(0, group_ids, torch.ones_like(values))
        means = sums / counts.clamp_min(1.0)
        return values / means[group_ids].clamp_min(1e-9)

    def _shuffle_dfl_per_gt(
        self,
        values: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Apply a vectorized deterministic random derangement inside every non-singleton GT group."""
        if not values.numel():
            return values
        group_ids, _ = self._positive_group_ids(fg_mask, target_gt_idx)
        step = int(self.dfl_shuffle_step.item())
        base_seed = int(self.dfl_shuffle_seed.item())
        seed = (base_seed + (step + 1) * 1_000_003 + (int(epoch) + 1) * 9_176) % (2**63 - 1)
        generator = torch.Generator(device=values.device)
        generator.manual_seed(seed)

        # First randomize positive indices, then use a stable group sort so that
        # every group's internal ordering stays random without a Python loop.
        random_order = torch.randperm(values.numel(), generator=generator, device=values.device)
        group_order = torch.argsort(group_ids[random_order], stable=True)
        order = random_order[group_order]
        ordered_groups = group_ids[order]
        positions = torch.arange(values.numel(), device=values.device)
        is_first = torch.ones_like(positions, dtype=torch.bool)
        is_first[1:] = ordered_groups[1:] != ordered_groups[:-1]
        is_last = torch.ones_like(positions, dtype=torch.bool)
        is_last[:-1] = ordered_groups[:-1] != ordered_groups[1:]
        group_starts = torch.cummax(torch.where(is_first, positions, 0), dim=0).values
        source_positions = torch.where(is_last, group_starts, positions + 1)
        source_indices = order[source_positions]
        shuffled = values.clone()
        shuffled[order] = values[source_indices]
        moved = int((source_indices != order).sum().item())
        with torch.no_grad():
            self.dfl_shuffle_step.add_(1)
        total = max(int(values.numel()), 1)
        self._last_dfl_shuffle_metrics = {
            "ftsc_dfl_shuffle_eligible_fraction": moved / total,
            "ftsc_dfl_shuffle_moved_fraction": moved / total,
            "ftsc_dfl_shuffle_step": float(step),
        }
        return shuffled

    def _gt_mass_factors(
        self,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        reference: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Return positive-only cls factors that moderately rebalance TAL support across GT groups.

        This is deliberately loss-only: it cannot create positives for an
        unassigned GT.  The inverse-support factor is bounded, normalized to
        mean one over all positives in the mini-batch, and enters through the
        same F5 residual schedule as the evidence gate.
        """
        group_ids, group_count = self._positive_group_ids(fg_mask, target_gt_idx)
        counts = reference.new_zeros(group_count).scatter_add_(0, group_ids, torch.ones_like(reference))
        active_group_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
        group_sizes = counts[active_group_ids]
        mean_support = group_sizes.mean()
        unclamped_group_factors = (mean_support / group_sizes.clamp_min(1.0)).pow(self.gt_mass_power)
        group_factors = unclamped_group_factors.clamp(self.gt_mass_min_factor, self.gt_mass_max_factor)

        moved_groups = 0
        shuffle_step = 0
        if self.gt_mass_shuffle and group_factors.numel() > 1:
            shuffle_step = int(self.gt_mass_shuffle_step.item())
            base_seed = int(self.gt_mass_shuffle_seed.item())
            seed = (base_seed + (shuffle_step + 1) * 1_000_003 + (int(epoch) + 1) * 9_176) % (2**63 - 1)
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(seed)
            order = torch.randperm(group_factors.numel(), generator=generator, device=reference.device)
            source = order.roll(1)
            shuffled = group_factors.clone()
            shuffled[order] = group_factors[source]
            group_factors = shuffled
            moved_groups = int(group_factors.numel())
            with torch.no_grad():
                self.gt_mass_shuffle_step.add_(1)

        factor_lookup = reference.new_ones(group_count)
        factor_lookup[active_group_ids] = group_factors
        positive_factors = factor_lookup[group_ids]
        positive_factors = positive_factors / positive_factors.mean().clamp_min(1e-9)
        rho = self.residual_fraction(epoch)
        scheduled_factors = 1.0 + rho * (positive_factors - 1.0)

        singleton_fraction = (group_sizes == 1).float().mean()
        self._last_gt_mass_metrics = {
            "ftsc_gt_group_count": float(group_sizes.numel()),
            "ftsc_mean_positives_per_gt": float(mean_support.detach().item()),
            "ftsc_single_positive_gt_fraction": float(singleton_fraction.detach().item()),
            "ftsc_gt_mass_factor_clamp_min_fraction": float(
                (unclamped_group_factors <= self.gt_mass_min_factor).float().mean().item()
            ),
            "ftsc_gt_mass_factor_clamp_max_fraction": float(
                (unclamped_group_factors >= self.gt_mass_max_factor).float().mean().item()
            ),
            "ftsc_gt_mass_shuffle": float(self.gt_mass_shuffle),
            "ftsc_gt_mass_shuffle_group_fraction": moved_groups / max(int(group_sizes.numel()), 1),
            "ftsc_gt_mass_shuffle_step": float(shuffle_step),
        }
        support = group_sizes.detach().float()
        factor = group_factors.detach().float()
        support_centered = support - support.mean()
        factor_centered = factor - factor.mean()
        denominator = support_centered.square().sum().sqrt() * factor_centered.square().sum().sqrt()
        self._last_gt_mass_metrics["ftsc_corr_support_count_gt_mass_factor"] = (
            float((support_centered * factor_centered).sum().div(denominator).item())
            if support.numel() > 1 and float(denominator.item()) > 0
            else 0.0
        )
        self._last_gt_mass_metrics.update(self._stats("ftsc_gt_support_count", group_sizes))
        self._last_gt_mass_metrics.update(self._stats("ftsc_gt_mass_factor_raw", positive_factors))
        self._last_gt_mass_metrics.update(self._stats("ftsc_gt_mass_factor_scheduled", scheduled_factors))
        return scheduled_factors

    def _gate(
        self,
        log_gate: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        clipped = log_gate.clamp(-self.log_clip, self.log_clip)
        raw_gate = clipped.exp()
        if self.per_gt_norm:
            # Log-centering controls evidence scale before clipping; this second
            # normalization prevents clipping from changing total GT loss mass.
            raw_gate = self._normalize_per_gt(raw_gate, fg_mask, target_gt_idx)
        rho = self.residual_fraction(epoch)
        return 1.0 + rho * (raw_gate - 1.0)

    @staticmethod
    def _stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
        if not values.numel():
            return {
                f"{prefix}_mean": 1.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 1.0,
                f"{prefix}_max": 1.0,
            }
        values = values.detach().float()
        return {
            f"{prefix}_mean": float(values.mean().item()),
            f"{prefix}_std": float(values.std(unbiased=False).item()),
            f"{prefix}_min": float(values.min().item()),
            f"{prefix}_max": float(values.max().item()),
        }

    def forward(
        self,
        anchor_points_px: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        pred_distri: torch.Tensor,
        epoch: int = 0,
        classification_quality: torch.Tensor | None = None,
        localization_quality: torch.Tensor | None = None,
        audit_target_scores: torch.Tensor | None = None,
        audit_stride: torch.Tensor | None = None,
        audit_anchor_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build positive-only classification, box and DFL weights after TAL assignment."""
        self.last_audit = {}
        positive_count = int(fg_mask.sum().item())
        if positive_count == 0:
            empty = pred_distri.new_empty(0)
            zero = pred_distri.sum() * 0.0
            self.last_metrics = {"ftsc_positive_count": 0.0, "ftsc_rho": self.residual_fraction(epoch)}
            if self.dfl_shuffle_within_gt:
                self.last_metrics.update(
                    {
                        "ftsc_dfl_shuffle_within_gt": 1.0,
                        "ftsc_dfl_shuffle_eligible_fraction": 0.0,
                        "ftsc_dfl_shuffle_moved_fraction": 0.0,
                        "ftsc_dfl_shuffle_step": float(self.dfl_shuffle_step.item()),
                    }
                )
            self.last_metrics.update(
                {
                    "ftsc_gt_group_count": 0.0,
                    "ftsc_mean_positives_per_gt": 0.0,
                    "ftsc_single_positive_gt_fraction": 0.0,
                    "ftsc_gt_mass_rebalance_cls": float(self.gt_mass_rebalance_cls),
                    "ftsc_gt_mass_shuffle": float(self.gt_mass_shuffle),
                }
            )
            if self.um_enabled:
                self.last_metrics.update(
                    {
                        "ftsc_um_lambda": self.um_lambda,
                        "ftsc_um_quality_guard": float(self.um_quality_guard),
                        "ftsc_um_quality_gamma": self.um_quality_gamma,
                        "ftsc_um_positive_count": 0.0,
                        "ftsc_um_positive_dfl_count": 0.0,
                        "ftsc_um_side_count": 0.0,
                        "ftsc_um_raw_loss": 0.0,
                        "ftsc_um_guarded_entropy_loss": 0.0,
                        "ftsc_um_guard_mass": 0.0,
                        "ftsc_um_guard_mean": 0.0,
                        "ftsc_um_guard_lt_025_fraction": 0.0,
                        "ftsc_um_guard_gt_075_fraction": 0.0,
                        "ftsc_um_weighted_loss": 0.0,
                    }
                )
                if self.um_quality_guard:
                    for prefix in ("ftsc_um_quality_iou", "ftsc_um_guard"):
                        self.last_metrics.update(
                            {f"{prefix}_{suffix}": 0.0 for suffix in ("mean", "std", "min", "max")}
                        )
            return {"cls": empty, "box": empty, "dfl": empty, "regularization": zero, "um_loss": zero}

        raw_evidence = {}
        centered = {}
        if "position_gaussian" in self.providers:
            position = self.providers["position_gaussian"](anchor_points_px, target_bboxes_px, fg_mask)
            raw_evidence["position_gaussian"] = position
            centered["position_gaussian"] = (
                self._center_per_gt(position, fg_mask, target_gt_idx) if self.per_gt_norm else position
            )
        if "fcos_centerness" in self.providers:
            centerness = self.providers["fcos_centerness"](anchor_points_px, target_bboxes_px, fg_mask)
            raw_evidence["fcos_centerness"] = centerness
            centered["fcos_centerness"] = (
                self._center_per_gt(centerness, fg_mask, target_gt_idx) if self.per_gt_norm else centerness
            )
        if "dcfl_dgmm_quality" in self.providers:
            if classification_quality is None or localization_quality is None:
                raise ValueError("DCFL-DGMM evidence requires detached classification and localization quality.")
            group_ids, group_count = self._positive_group_ids(fg_mask, target_gt_idx)
            quality_evidence = self.providers["dcfl_dgmm_quality"](
                classification_quality, localization_quality, group_ids, group_count
            )
            raw_evidence["dcfl_dgmm_quality"] = quality_evidence
            centered["dcfl_dgmm_quality"] = (
                self._center_per_gt(quality_evidence, fg_mask, target_gt_idx)
                if self.per_gt_norm
                else quality_evidence
            )
        if "dfl_distribution" in self.providers:
            distribution = self.providers["dfl_distribution"](pred_distri, fg_mask)
            if self.dfl_shuffle_within_gt:
                distribution = self._shuffle_dfl_per_gt(distribution, fg_mask, target_gt_idx, epoch)
            raw_evidence["dfl_distribution"] = distribution
            centered["dfl_distribution"] = (
                self._center_per_gt(distribution, fg_mask, target_gt_idx) if self.per_gt_norm else distribution
            )

        reference = next(iter(centered.values()))
        zero = torch.zeros_like(reference)
        task_logs = {"cls": zero.clone(), "box": zero.clone(), "dfl": zero.clone()}
        for name, values in centered.items():
            if name in self.POSITION_EVIDENCE:
                for task in self.position_tasks:
                    strength_task = task if self.position_task_specific_strength else None
                    task_logs[task] = task_logs[task] + self.strength(name, values, task=strength_task) * values
            else:
                contribution = self.strength(name, values) * values
                if self.dfl_apply_cls:
                    task_logs["cls"] = task_logs["cls"] + contribution
                if self.dfl_apply_box:
                    task_logs["box"] = task_logs["box"] + contribution
                if self.dfl_apply_dfl:
                    task_logs["dfl"] = task_logs["dfl"] + contribution

        weights = {}
        for task, values in task_logs.items():
            weights[task] = self._gate(values, fg_mask, target_gt_idx, epoch)
        intra_gt_cls_weight = weights["cls"]

        # This branch changes only positive target-class loss mass. It does not
        # alter TAL assignment, negative/off-class weights, box, or DFL losses.
        if self.gt_mass_rebalance_cls:
            weights["cls"] = weights["cls"] * self._gt_mass_factors(
                fg_mask, target_gt_idx, reference, epoch
            )

        regularization = self._strength_regularization(reference)
        um_loss = (
            self.um_regularizer(pred_distri, fg_mask, localization_quality=localization_quality)
            if self.um_regularizer is not None
            else pred_distri.sum() * 0.0
        )

        metrics = {
            "ftsc_positive_count": float(positive_count),
            "ftsc_rho": self.residual_fraction(epoch),
            "ftsc_negative_weight": 1.0,
        }
        for name, values in centered.items():
            metrics.update(self._stats(f"ftsc_{name}", values))
            if name in self.POSITION_EVIDENCE and self.position_task_specific_strength:
                task_strengths = []
                for task in self.position_tasks:
                    task_strength = self.strength(name, reference, task=task).detach()
                    task_strengths.append(task_strength)
                    metrics[f"ftsc_strength_{name}_{task}"] = float(task_strength.item())
                metrics[f"ftsc_strength_{name}"] = float(torch.stack(task_strengths).mean().item())
            else:
                metrics[f"ftsc_strength_{name}"] = float(self.strength(name, reference).detach().item())
        for task, values in weights.items():
            metrics.update(self._stats(f"ftsc_weight_{task}", values))
            metrics[f"ftsc_{task}_clamp_low_fraction"] = float(
                (task_logs[task] < -self.log_clip).float().mean().item()
            )
            metrics[f"ftsc_{task}_clamp_high_fraction"] = float(
                (task_logs[task] > self.log_clip).float().mean().item()
            )
        metrics.update(self._stats("ftsc_intra_gt_cls_weight", intra_gt_cls_weight))
        metrics.update(self._stats("ftsc_effective_positive_cls_multiplier", weights["cls"]))
        metrics["ftsc_positive_negative_weight_ratio"] = float(weights["cls"].detach().float().mean().item())
        distribution_provider = self.providers["dfl_distribution"] if "dfl_distribution" in self.providers else None
        if distribution_provider is not None:
            metrics.update(self._stats("ftsc_dfl_entropy", distribution_provider.last_entropy))
            metrics.update(self._stats("ftsc_dfl_variance", distribution_provider.last_variance))
        dcfl_provider = self.providers["dcfl_dgmm_quality"] if "dcfl_dgmm_quality" in self.providers else None
        if dcfl_provider is not None:
            metrics.update(dcfl_provider.last_metrics)
        if self.um_regularizer is not None:
            metrics.update(self._stats("ftsc_um_entropy_raw", self.um_regularizer.last_raw_entropy))
            metrics.update(self._stats("ftsc_um_entropy_normalized", self.um_regularizer.last_normalized_entropy))
            metrics.update(
                {
                    "ftsc_um_lambda": self.um_lambda,
                    "ftsc_um_quality_guard": float(self.um_quality_guard),
                    "ftsc_um_quality_gamma": self.um_quality_gamma,
                    "ftsc_um_positive_count": float(positive_count),
                    "ftsc_um_positive_dfl_count": float(positive_count),
                    "ftsc_um_side_count": float(positive_count * 4),
                    "ftsc_um_raw_loss": float(um_loss.detach().item()),
                    "ftsc_um_guarded_entropy_loss": float(um_loss.detach().item()),
                    "ftsc_um_weighted_loss": float((self.um_lambda * um_loss.detach()).item()),
                }
            )
            if self.um_quality_guard:
                quality = self.um_regularizer.last_quality
                guard = self.um_regularizer.last_guard
                metrics.update(self._stats("ftsc_um_quality_iou", quality))
                metrics.update(self._stats("ftsc_um_guard", guard))
                metrics.update(
                    {
                        "ftsc_um_guard_mass": float(guard.sum().item()),
                        "ftsc_um_guard_mean": float(guard.mean().item()),
                        "ftsc_um_guard_lt_025_fraction": float((guard < 0.25).float().mean().item()),
                        "ftsc_um_guard_gt_075_fraction": float((guard > 0.75).float().mean().item()),
                    }
                )
        metrics["ftsc_position_task_specific_strength"] = float(self.position_task_specific_strength)
        metrics["ftsc_dfl_shuffle_within_gt"] = float(self.dfl_shuffle_within_gt)
        metrics["ftsc_gt_mass_rebalance_cls"] = float(self.gt_mass_rebalance_cls)
        if self.dfl_shuffle_within_gt:
            metrics.update(self._last_dfl_shuffle_metrics)
        if self.gt_mass_rebalance_cls:
            metrics.update(self._last_gt_mass_metrics)
        else:
            group_ids, group_count = self._positive_group_ids(fg_mask, target_gt_idx)
            counts = reference.new_zeros(group_count).scatter_add_(0, group_ids, torch.ones_like(reference))
            active_counts = counts[counts > 0]
            metrics.update(
                {
                    "ftsc_gt_group_count": float(active_counts.numel()),
                    "ftsc_mean_positives_per_gt": float(active_counts.mean().item()),
                    "ftsc_single_positive_gt_fraction": float((active_counts == 1).float().mean().item()),
                    "ftsc_gt_mass_shuffle": 0.0,
                }
            )
        self.last_metrics = metrics
        if self.audit_enabled:
            if audit_target_scores is None or audit_stride is None or audit_anchor_index is None:
                raise ValueError("FTSC audit requires target scores, stride labels, and anchor indices.")
            distribution_provider = self.providers["dfl_distribution"] if "dfl_distribution" in self.providers else None
            self.last_audit = self._make_audit_snapshot(
                fg_mask=fg_mask,
                target_gt_idx=target_gt_idx,
                target_scores=audit_target_scores,
                stride=audit_stride,
                anchor_index=audit_anchor_index,
                audit_decoded_iou=localization_quality,
                raw_evidence=raw_evidence,
                task_logs=task_logs,
                weights=weights,
                dfl_entropy=(distribution_provider.last_entropy if distribution_provider is not None else None),
                dfl_variance=(distribution_provider.last_variance if distribution_provider is not None else None),
                epoch=epoch,
            )
        return {**weights, "regularization": regularization, "um_loss": um_loss}


class FTSCFeatureCalibrator(nn.Module):
    """Identity-initialized P2 detail calibrator for tiny object detection.

    The module estimates local high-frequency evidence from a feature map and
    uses it to build a gentle spatial-channel gate:

        detail = |x - AvgPool(x)|
        z      = detail / mean(detail)
        out    = x * (1 + alpha * (gate - 1))

    `alpha` is initialized near zero, so the module starts close to identity and
    lets training decide whether the tiny-detail cue is useful.
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.25,
        detail_kernel: int = 3,
        alpha_init: float = 1e-3,
        alpha_max: float = 0.35,
        spatial_temperature: float = 1.0,
        use_channel_gate: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.detail_kernel = max(int(detail_kernel) | 1, 3)
        self.alpha_max = max(float(alpha_max), 0.0)
        self.spatial_temperature = float(spatial_temperature)
        self.use_channel_gate = bool(use_channel_gate)

        hidden = max(int(self.channels * float(hidden_ratio)), 8)
        self.channel_gate = (
            nn.Sequential(
                nn.Conv2d(self.channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, self.channels, kernel_size=1),
            )
            if self.use_channel_gate
            else None
        )

        p = max(min(float(alpha_init) / max(self.alpha_max, 1e-12), 1.0 - 1e-6), 1e-6)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(p)))
        self.spatial_bias = nn.Parameter(torch.zeros(1))
        self.last_gate: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        """Return the bounded residual gate strength."""
        return torch.sigmoid(self.alpha_logit) * self.alpha_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply detail-aware feature calibration."""
        detail = (x - F.avg_pool2d(x, self.detail_kernel, stride=1, padding=self.detail_kernel // 2)).abs()
        energy = detail.mean(dim=1, keepdim=True)
        normalizer = energy.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        spatial_gate = torch.sigmoid(self.spatial_temperature * (energy / normalizer - 1.0) - self.spatial_bias)

        if self.channel_gate is not None:
            channel_gate = torch.sigmoid(
                self.channel_gate(F.adaptive_avg_pool2d(detail, 1))
                + self.channel_gate(F.adaptive_max_pool2d(detail, 1))
            )
            gate = 1.0 + spatial_gate * channel_gate
        else:
            gate = 1.0 + spatial_gate

        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        self.last_gate = gate.detach() if self.training else None
        return x * (1.0 + alpha * (gate - 1.0))
