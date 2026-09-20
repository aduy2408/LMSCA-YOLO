# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import time

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.ops import xywh2xyxy


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
    nms_method: str = "hard",
    soft_nms_sigma: float = 0.5,
    soft_nms_min_score: float = 0.001,
    box_voting: bool = False,
    box_voting_iou: float = 0.5,
    box_voting_weight: str = "score_iou",
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection
    formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each box can have multiple labels.
        labels (list[torch.Tensor]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        nc (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for NMS.
        max_wh (int): Maximum box width and height in pixels.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS.
        return_idxs (bool): Whether to return the indices of kept detections.
        nms_method (str): NMS method to use: "hard", "soft-linear", or "soft-gaussian".
        soft_nms_sigma (float): Sigma value for Gaussian Soft-NMS.
        soft_nms_min_score (float): Minimum score to keep after Soft-NMS decay.
        box_voting (bool): Whether to refine kept box coordinates using overlapping pre-NMS boxes.
        box_voting_iou (float): Minimum IoU for boxes that vote on a kept box.
        box_voting_weight (str): Box voting weight mode, either "score_iou" or "score_iou2".

    Returns:
        (list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]): List of detections per image with shape
            (num_boxes, 6 + num_masks) containing (x1, y1, x2, y2, confidence, class, mask1, mask2, ...). If
            return_idxs=True, returns a tuple of (output, keepi) where keepi contains indices of kept detections.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= box_voting_iou <= 1, (
        f"Invalid box voting IoU {box_voting_iou}, valid values are between 0.0 and 1.0"
    )
    assert nms_method in {"hard", "soft-linear", "soft-gaussian"}, (
        f"Invalid NMS method {nms_method}, valid values are 'hard', 'soft-linear', or 'soft-gaussian'"
    )
    assert box_voting_weight in {"score_iou", "score_iou2"}, (
        f"Invalid box voting weight {box_voting_weight}, valid values are 'score_iou' or 'score_iou2'"
    )
    assert soft_nms_sigma > 0, f"Invalid Soft-NMS sigma {soft_nms_sigma}, valid values are greater than 0.0"
    assert 0 <= soft_nms_min_score <= 1, (
        f"Invalid Soft-NMS minimum score {soft_nms_min_score}, valid values are between 0.0 and 1.0"
    )
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation mode, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            if nms_method != "hard":
                i = TorchNMS.soft_nms(
                    boxes,
                    scores,
                    iou_thres,
                    method=nms_method[5:],
                    sigma=soft_nms_sigma,
                    min_score=soft_nms_min_score,
                )
            else:
                # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
                if "torchvision" in sys.modules:
                    import torchvision  # scope as slow import

                    i = torchvision.ops.nms(boxes, scores, iou_thres)
                else:
                    i = TorchNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if box_voting and not rotated and i.numel():
            output[xi][:, :4] = TorchNMS.box_voting(
                x[:, :4],
                x[:, 4],
                x[:, 5],
                i,
                box_voting_iou,
                box_voting_weight,
            )
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output


class TorchNMS:
    """Ultralytics custom NMS implementation optimized for YOLO.

    This class provides static methods for performing non-maximum suppression (NMS) operations on bounding boxes,
    including standard NMS, fast NMS, and batched NMS for multi-class scenarios.

    Methods:
        fast_nms: Fast-NMS using upper triangular matrix operations.
        nms: Optimized NMS with early termination that matches torchvision behavior exactly.
        batched_nms: Batched NMS for class-aware suppression.

    Examples:
        Perform standard NMS on boxes and scores
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = TorchNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def box_voting(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        keep: torch.Tensor,
        iou_threshold: float = 0.5,
        weight_mode: str = "score_iou",
    ) -> torch.Tensor:
        """Refine kept box coordinates as score/IoU-weighted averages of same-class pre-NMS boxes."""
        if keep.numel() == 0:
            return boxes.new_zeros((0, 4))

        kept_boxes = boxes[keep]
        refined = kept_boxes.clone()
        ious = box_iou(kept_boxes, boxes)
        for row in range(keep.numel()):
            same_class = classes == classes[keep[row]]
            mask = same_class & (ious[row] >= iou_threshold)
            weights = scores[mask] * ious[row, mask]
            if weight_mode == "score_iou2":
                weights = weights * ious[row, mask]
            denom = weights.sum()
            if denom > 0:
                refined[row] = (weights[:, None] * boxes[mask]).sum(0) / denom
        return refined

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        """Fast-NMS implementation from https://arxiv.org/pdf/1904.02689 using upper triangular matrix operations.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_triu (bool): Whether to use torch.triu operator for upper triangular matrix operations.
            iou_func (callable): Function to compute IoU between boxes.
            exit_early (bool): Whether to exit early if there are no boxes.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.fast_nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # Zeroing these scores ensures the additional indices would not affect the final results
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_  # update original tensor for NMSModel
            # NOTE: return indices with fixed length to avoid TFLite reshape error
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Pre-allocate and extract coordinates once
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores descending
        order = scores.argsort(0, descending=True)

        # Pre-allocate keep list with maximum possible size
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            # Vectorized IoU calculation for remaining boxes
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            # Fast intersection and IoU
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            # Early exit: skip IoU calculation if no intersection
            if inter.sum() == 0:
                # No overlaps with current box, keep all remaining boxes
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            # Keep boxes with IoU <= threshold
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def soft_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        method: str = "linear",
        sigma: float = 0.5,
        min_score: float = 0.001,
    ) -> torch.Tensor:
        """Soft-NMS that decays overlapping boxes and returns kept indices ordered by final selection.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,). The tensor is updated in-place with decayed scores.
            iou_threshold (float): IoU threshold used by linear Soft-NMS.
            method (str): Soft-NMS method, either "linear" or "gaussian".
            sigma (float): Sigma value for Gaussian Soft-NMS.
            min_score (float): Minimum score to keep after score decay.

        Returns:
            (torch.Tensor): Indices of boxes to keep after Soft-NMS.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        if method not in {"linear", "gaussian"}:
            raise ValueError(f"Invalid Soft-NMS method '{method}'. Expected 'linear' or 'gaussian'.")
        if sigma <= 0:
            raise ValueError(f"Invalid Soft-NMS sigma {sigma}. Expected a value greater than 0.0.")

        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        idxs = torch.arange(scores.numel(), device=boxes.device)
        keep = torch.zeros(scores.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0

        while idxs.numel() > 0:
            max_pos = scores[idxs].argmax()
            current = idxs[max_pos]
            current_score = scores[current]
            if current_score < min_score:
                break

            keep[keep_idx] = current
            keep_idx += 1
            idxs = idxs[idxs != current]
            if idxs.numel() == 0:
                break

            xx1 = torch.maximum(x1[current], x1[idxs])
            yy1 = torch.maximum(y1[current], y1[idxs])
            xx2 = torch.minimum(x2[current], x2[idxs])
            yy2 = torch.minimum(y2[current], y2[idxs])
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            iou = inter / (areas[current] + areas[idxs] - inter)

            if method == "linear":
                decay = torch.ones_like(iou)
                decay = torch.where(iou > iou_threshold, 1 - iou, decay)
            else:
                decay = torch.exp(-((iou * iou) / sigma))
            scores[idxs] *= decay
            idxs = idxs[scores[idxs] >= min_score]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """Batched NMS for class-aware suppression.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            idxs (torch.Tensor): Class indices with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_fast_nms (bool): Whether to use the Fast-NMS implementation.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply batched NMS across multiple classes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> idxs = torch.tensor([0, 1])
            >>> keep = TorchNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Strategy: offset boxes by class index to prevent cross-class suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )
