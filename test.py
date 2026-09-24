import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.config import (
    CAMUS_SPLIT,
    CAMUS_TASK,
    CAMUS_TASKS,
    DEFAULT_CAMUS_DATA_PATH,
    DEFAULT_ECHONET_DATA_PATH,
    TASKS,
    FinalConfig,
    resolve_data_settings,
)
from utils.data import CardiacVideoDataset
from utils.engine import evaluate, load_checkpoint_strict, seed_everything
from models import model_dict


def arguments():
    parser = argparse.ArgumentParser(description="Test the final CAMUS/EchoNet E3SAM2 checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS, default=CAMUS_TASK)
    parser.add_argument("--data", type=Path, help="Optional override for the task-selected data root")
    parser.add_argument("--camus_data_path", "--camus-data-path", type=Path, default=DEFAULT_CAMUS_DATA_PATH)
    parser.add_argument("--echo_data_path", "--echo-data-path", type=Path, default=DEFAULT_ECHONET_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for test metrics and optional masks (default: results)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--mask-loss", choices=("bce_dice", "structure"), default="bce_dice",
        help="Mask loss used when reporting test loss",
    )
    parser.add_argument("--check-only", action="store_true", help="Only build the model and validate all checkpoint keys")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument(
        "--compute_ef", "--compute-ef", action="store_true",
        help="Compute CAMUS biplane EF/EDV/ESV or EchoNet PCA single-plane EF (test_npz protocol)",
    )
    return parser.parse_args()


def main():
    args = arguments()
    if args.compute_ef and not args.check_only:
        from utils.cardiac_metrics import require_ef_dependencies
        require_ef_dependencies()
    data_path, image_size = resolve_data_settings(
        args.task,
        args.camus_data_path,
        args.echo_data_path,
        args.data,
    )
    config = FinalConfig(
        data_path,
        args.output,
        task=args.task,
        image_size=image_size,
        batch_size=1,
        num_workers=args.workers,
        mask_loss=args.mask_loss,
    )
    seed_everything(config.seed)
    device = torch.device(args.device)
    model = model_dict.build_model(config).to(device)
    tensor_count = load_checkpoint_strict(model, args.checkpoint)
    print(f"Strict checkpoint check passed: {tensor_count} tensors")
    if args.check_only:
        return

    print(f"Task: {config.task}, image size: {config.image_size}")
    if config.task in CAMUS_TASKS:
        print(f"CAMUS split: {CAMUS_SPLIT}")
    print(f"Data path: {config.data_path}")
    dataset = CardiacVideoDataset(
        config.data_path, config.test_split, config.image_size, config.frame_length,
        training=False,
    )
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    metrics = evaluate(
        model, loader, device, config, detailed=True,
        mask_output_dir=args.output / "masks" if args.save_masks else None,
        compute_ef=args.compute_ef,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n" + "=" * 40)
    print(f"Results for {args.checkpoint}")
    for title, metric_key in (
        ("All-frame metrics", "all_frame"),
        ("ED/ES-only metrics", "ed_es_only"),
    ):
        values = metrics[metric_key]
        print(f"\n--- {title} ---")
        print("dice_mean    iou_mean    hd_mean    assd_mean")
        print(values["dice_mean"], values["iou_mean"], values["hd_mean"], values["assd_mean"])
        print("dices_std    iou_std    hd_std    assd_std")
        print(values["dices_std"], values["iou_std"], values["hd_std"], values["assd_std"])
    if args.compute_ef:
        cardiac = metrics["cardiac"]
        print(f"\nCardiac metrics: {cardiac['patients_evaluated']}/{cardiac['patients_total']} patients, "
              f"{cardiac['patients_skipped']} skipped")
        print("Method:", cardiac["method"])
        if "note" in cardiac:
            print(cardiac["note"])
        for record in cardiac["patients"]:
            prefix = "video_name" if config.task == "EchoNet_Video" else "patient_name"
            print(f"{prefix}{record['patient']}---pred_efs{record['pred']['ef']}---gt_efs{record['gt']['ef']}")
        for key in cardiac["metric_names"]:
            values = cardiac[key]
            print(f"{key.upper()} [{cardiac['units'][key]}] "
                  f"MAE: {values['mae']}, bias(pred-gt): {values['bias']}, "
                  f"std(pred-gt): {values['std']}, corr: {values['corr']}")
            print("wilcoxon_signed_rank_test:", values["wilcoxon_signed_rank_test"])
        for skipped in cardiac["skipped"]:
            print(f"Skipped {skipped['patient']}: {skipped['reason']}")
        for failure in cardiac["failures"]:
            print(f"EF=0 fallback for {failure['patient']}: {failure['reason']}")
    print(f"Saved metrics to {args.output / 'metrics.json'}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
