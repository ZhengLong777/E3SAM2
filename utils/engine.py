from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .losses import EdgeBCELoss, build_mask_loss, entropy_mask_regularization


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def reset_memory(model):
    """Explicit reset for callers; train20 training/evaluation never resets per batch."""
    base = unwrap(model)
    if hasattr(base, "memory_bank_list"):
        base.memory_bank_list.clear()


def load_checkpoint_strict(model, path: str | Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(checkpoint)
        if tuple(expected[key].shape) != tuple(checkpoint[key].shape)
    )
    if missing or unexpected or mismatched:
        report = {"missing": missing, "unexpected": unexpected, "shape_mismatch": mismatched}
        raise RuntimeError("Checkpoint is not strictly compatible:\n" + json.dumps(report, indent=2))
    model.load_state_dict(checkpoint, strict=True)
    return len(checkpoint)


def make_prompt(batch, device):
    return batch["points"].float().to(device), batch["point_labels"].long().to(device)


def segmentation_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float):
    """Return per-frame [B, T] counts; do not pool the temporal dimension."""
    prediction = torch.sigmoid(logits) > threshold
    target = target > 0.5
    dims = (-2, -1)
    tp = (prediction & target).sum(dims).double()
    fp = (prediction & ~target).sum(dims).double()
    fn = (~prediction & target).sum(dims).double()
    return tp, fp, fn


def summarize_segmentation_metrics(dices, ious, hd95_values, assd_values):
    return {
        "dice_mean": float(np.mean(dices)),
        "iou_mean": float(np.mean(ious)),
        "hd_mean": float(np.mean(hd95_values)) if hd95_values else float("nan"),
        "assd_mean": float(np.mean(assd_values)) if assd_values else float("nan"),
        "dices_std": float(np.std(dices)),
        "iou_std": float(np.std(ious)),
        "hd_std": float(np.std(hd95_values)) if hd95_values else float("nan"),
        "assd_std": float(np.std(assd_values)) if assd_values else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, device, config, detailed: bool = False,
             mask_output_dir: str | Path | None = None, compute_ef: bool = False):
    cardiac = None
    if compute_ef:
        if not detailed:
            raise ValueError("EF calculation requires detailed testing")
        from .cardiac_metrics import CardiacMetrics
        cardiac = CardiacMetrics(config.task)
    model.eval()
    mask_loss = build_mask_loss(config.mask_loss, config.positive_weight).to(device)
    tp_all, fp_all, fn_all = [], [], []
    frame_dices, frame_ious = [], []
    hd95_values, assd_values = [], []
    ed_es_dices, ed_es_ious = [], []
    ed_es_hd95_values, ed_es_assd_values = [], []
    losses = []
    indices = list(config.evaluation_frames)
    if mask_output_dir is not None:
        mask_output_dir = Path(mask_output_dir)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
    if detailed:
        try:
            from medpy.metric.binary import assd as medpy_assd
            from medpy.metric.binary import hd95 as medpy_hd95
        except ImportError as error:
            raise RuntimeError(
                "Detailed test metrics require MedPy. Install the project requirements first."
            ) from error
    for batch in loader:
        images = batch["image"].float().to(device)
        targets = batch["label"].float().to(device)
        # Preserve the model's bank across batches and train/validation, as in train20.
        masks, _ = model(images, make_prompt(batch, device), None)
        if mask_output_dir is not None:
            # Save this exact forward pass: a second pass would use different memory/RNG.
            saved_masks = (torch.sigmoid(masks[:, :, 0]) > config.prediction_threshold).cpu().numpy()
            for name, sample_masks in zip(batch["image_name"], saved_masks):
                np.savez_compressed(
                    mask_output_dir / f"{Path(name).stem}.npz", masks=sample_masks.astype(np.uint8)
                )
        logits = masks[:, indices, 0]
        targets = targets[:, indices]
        losses.append(mask_loss(logits, targets).item())
        tp, fp, fn = segmentation_counts(logits, targets, config.prediction_threshold)
        tp_all.append(tp.cpu())
        fp_all.append(fp.cpu())
        fn_all.append(fn.cpu())
        if detailed:
            predictions = (torch.sigmoid(logits) > config.prediction_threshold).cpu().numpy()
            target_masks = (targets > 0.5).cpu().numpy()
            spacing_batch = batch["spacing"].detach().cpu().numpy()
            if cardiac is not None:
                metadata = {key: batch[key].detach().cpu().numpy().reshape(-1)
                            for key in ("ef", "edv", "esv")}
                for sample_index, name in enumerate(batch["image_name"]):
                    cardiac.add_sample(
                        name, predictions[sample_index], spacing_batch[sample_index],
                        **{key: values[sample_index] for key, values in metadata.items()},
                    )
            for sample_index in range(predictions.shape[0]):
                spacing_values = np.asarray(spacing_batch[sample_index]).reshape(-1)
                voxel_spacing = spacing_values[:2][::-1] if spacing_values.size >= 2 else None
                for frame_index in range(predictions.shape[1]):
                    prediction = predictions[sample_index, frame_index]
                    target = target_masks[sample_index, frame_index]
                    is_ed_es = frame_index in (0, predictions.shape[1] - 1)
                    frame_tp = np.logical_and(prediction, target).sum(dtype=np.float64)
                    frame_fp = np.logical_and(prediction, np.logical_not(target)).sum(dtype=np.float64)
                    frame_fn = np.logical_and(np.logical_not(prediction), target).sum(dtype=np.float64)
                    frame_dice = (
                        (2.0 * frame_tp + 1e-5) / (2.0 * frame_tp + frame_fp + frame_fn + 1e-5)
                    )
                    frame_iou = (frame_tp + 1e-5) / (frame_tp + frame_fp + frame_fn + 1e-5)
                    frame_dices.append(frame_dice)
                    frame_ious.append(frame_iou)
                    if is_ed_es:
                        ed_es_dices.append(frame_dice)
                        ed_es_ious.append(frame_iou)
                    if prediction.any() and target.any():
                        hd95 = float(medpy_hd95(prediction, target, voxelspacing=voxel_spacing))
                        assd = float(medpy_assd(prediction, target, voxelspacing=voxel_spacing))
                        hd95_values.append(hd95)
                        assd_values.append(assd)
                        if is_ed_es:
                            ed_es_hd95_values.append(hd95)
                            ed_es_assd_values.append(assd)
    if detailed:
        result = {
            "all_frame": summarize_segmentation_metrics(
                frame_dices, frame_ious, hd95_values, assd_values
            ),
            "ed_es_only": summarize_segmentation_metrics(
                ed_es_dices, ed_es_ious, ed_es_hd95_values, ed_es_assd_values
            ),
        }
        if cardiac is not None:
            result["cardiac"] = cardiac.compute()
        return result
    tp, fp, fn = torch.cat(tp_all), torch.cat(fp_all), torch.cat(fn_all)
    dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
    iou = (tp + 1e-5) / (tp + fp + fn + 1e-5)
    return {"loss": float(np.mean(losses)), "dice": float(dice.mean()), "iou": float(iou.mean())}


def train_one_epoch(model, loader, optimizer, device, config, epoch, iteration, max_iterations):
    model.train()
    mask_loss_fn = build_mask_loss(config.mask_loss, config.positive_weight).to(device)
    edge_loss_fn = EdgeBCELoss(config.positive_weight).to(device)
    totals = []
    indices = list(config.supervised_frames)
    for batch in loader:
        images = batch["image"].float().to(device)
        masks = batch["label"].float().to(device)
        edges = batch["edge"].float().to(device)
        # Match train20: keep historical memory, including across epoch boundaries.
        optimizer.zero_grad(set_to_none=True)
        mask_logits, edge_logits = model(images, make_prompt(batch, device), None)
        mask_loss = mask_loss_fn(mask_logits[:, indices, 0], masks[:, indices])
        edge_loss = edge_loss_fn(edge_logits[:, indices, 0], edges[:, indices])
        supervised = config.mask_loss_weight * mask_loss + config.edge_loss_weight * edge_loss
        if epoch >= config.regularization_start_epoch:
            regularization = entropy_mask_regularization(masks, config.supervised_frames)
        else:
            from models.sam2_train.modeling.backbones.hieradet import EntropyAttention
            EntropyAttention.global_attention_maps.clear()
            regularization = torch.zeros((), device=device)
        total = supervised + config.regularization_weight * regularization
        total.backward()
        optimizer.step()

        if iteration < config.warmup_iterations:
            learning_rate = config.base_lr * (iteration + 1) / config.warmup_iterations
        else:
            shifted = iteration - config.warmup_iterations
            learning_rate = config.base_lr * (1.0 - shifted / max_iterations) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        iteration += 1
        totals.append([total.item(), mask_loss.item(), edge_loss.item(), regularization.item()])
    means = np.asarray(totals).mean(axis=0)
    return iteration, {"total": means[0], "mask": means[1], "edge": means[2], "regularization": means[3]}
