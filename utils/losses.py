import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam2_train.modeling.backbones.hieradet import EntropyAttention


class DiceBCELoss(nn.Module):
    def __init__(self, positive_weight: float = 2.0, dice_weight: float = 0.8):
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.positive_weight)
        probability = torch.sigmoid(logits)
        intersection = torch.sum(probability * target)
        dice = 1.0 - (2.0 * intersection + 1e-5) / (
            torch.sum(probability.square()) + torch.sum(target.square()) + 1e-5
        )
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice


class StructureLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        weighted_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weighted_bce = (weight * weighted_bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        probability = torch.sigmoid(logits)
        intersection = ((probability * target) * weight).sum(dim=(2, 3))
        union = ((probability + target) * weight).sum(dim=(2, 3))
        weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
        return (weighted_bce + weighted_iou).mean()


def build_mask_loss(name: str, positive_weight: float = 2.0) -> nn.Module:
    if name == "bce_dice":
        return DiceBCELoss(positive_weight)
    if name == "structure":
        return StructureLoss()
    raise ValueError(f"Unsupported mask loss: {name}")


class EdgeBCELoss(nn.Module):
    def __init__(self, positive_weight: float = 2.0):
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.positive_weight)


def entropy_mask_regularization(masks: torch.Tensor, supervised_frames=(0, -1)) -> torch.Tensor:
    batch, target_frames, _, _ = masks.shape
    target_indices = [frame if frame >= 0 else target_frames + frame for frame in supervised_frames]
    selected_masks = masks[:, target_indices].flatten(0, 1)
    total = torch.zeros((), device=masks.device)
    count = 0
    for _, entropy in EntropyAttention.global_attention_maps:
        side = int(entropy.shape[-1] ** 0.5)
        prediction_frames = entropy.shape[0] // batch
        prediction_indices = [
            frame if frame >= 0 else prediction_frames + frame for frame in supervised_frames
        ]
        selected_entropy = entropy.view(batch, prediction_frames, side, side)[:, prediction_indices].flatten(0, 1)
        resized_masks = nn.functional.interpolate(
            selected_masks[:, None].float(), size=(side, side), mode="bilinear", align_corners=False
        ).squeeze(1)
        inside = (selected_entropy * resized_masks).sum((1, 2)) / (resized_masks.sum((1, 2)) + 1e-6)
        outside_mask = 1.0 - resized_masks
        outside = (selected_entropy * outside_mask).sum((1, 2)) / (outside_mask.sum((1, 2)) + 1e-6)
        total = total - (outside - inside).clamp(min=1e-3, max=1.0).mean()
        count += 1
    EntropyAttention.global_attention_maps.clear()
    return total / count if count else total
