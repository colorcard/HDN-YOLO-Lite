"""NWD-based bounding-box regression for HDN-YOLO-Lite.

Implements the regression term in Section 3.4, Eqs. (13)-(16). The default
values follow the paper's training setup (``r_s = 0.035``, ``C = 12.16``).

The regression term is

    L_box = (1 - r_s) * L_CIoU + r_s * (1 - NWD),

where NWD is the normalized Gaussian Wasserstein similarity between the
predicted and ground-truth boxes. Classification, label assignment, NMS, and
DFL are left unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

__all__ = [
    "normalized_wasserstein",
    "NWDBboxLoss",
    "install_nwd_loss",
]


def normalized_wasserstein(
    pred: torch.Tensor,
    target: torch.Tensor,
    constant: float = 12.16,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the NWD similarity for axis-aligned boxes in ``xyxy`` format.

    Both tensors must already be expressed in the same normalized coordinate
    domain used by the regression branch (the domain in which the paper uses
    ``C = 12.16``).
    """
    center1 = (pred[:, :2] + pred[:, 2:]) / 2
    center2 = (target[:, :2] + target[:, 2:]) / 2
    center_delta = center1 - center2
    center_distance = center_delta[:, 0].square() + center_delta[:, 1].square() + eps

    w1 = pred[:, 2] - pred[:, 0] + eps
    h1 = pred[:, 3] - pred[:, 1] + eps
    w2 = target[:, 2] - target[:, 0] + eps
    h2 = target[:, 3] - target[:, 1] + eps
    wh_distance = ((w1 - w2).square() + (h1 - h2).square()) / 4

    wasserstein = torch.sqrt(center_distance + wh_distance)
    return torch.exp(-wasserstein / constant)


class NWDBboxLoss(BboxLoss):
    """CIoU + NWD mixed bounding-box loss used by HDN-YOLO-Lite."""

    #: NWD mixing weight (paper: r_s = 0.035).
    nwd_ratio: float = 0.035
    #: NWD normalization constant (paper: C = 12.16).
    nwd_constant: float = 12.16

    def __init__(self, reg_max: int = 16, *args, **kwargs):
        # Extra keyword arguments are accepted so the class also works with
        # Ultralytics forks that pass additional BboxLoss options.
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
        current_epoch: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou_term = 1.0 - iou

        nwd = normalized_wasserstein(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            constant=self.nwd_constant,
        ).unsqueeze(-1)
        nwd = torch.nan_to_num(nwd, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)

        loss_iou_term = (1.0 - self.nwd_ratio) * loss_iou_term + self.nwd_ratio * (1.0 - nwd)
        loss_iou_term = torch.nan_to_num(loss_iou_term, nan=1.0, posinf=1.0, neginf=1.0)
        loss_iou = (loss_iou_term * weight).sum() / target_scores_sum

        # Distribution Focal Loss is unchanged from YOLO11.
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
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


def install_nwd_loss(ratio: float = 0.035, constant: float = 12.16) -> type[NWDBboxLoss]:
    """Activate the NWD box loss for subsequent Ultralytics training.

    ``v8DetectionLoss`` instantiates ``ultralytics.utils.loss.BboxLoss`` at
    runtime, so replacing that symbol is sufficient and keeps the installed
    Ultralytics package untouched.
    """
    from ultralytics.utils import loss as loss_module

    NWDBboxLoss.nwd_ratio = float(ratio)
    NWDBboxLoss.nwd_constant = float(constant)
    loss_module.BboxLoss = NWDBboxLoss
    return NWDBboxLoss
