import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def configure_visible_gpus():
    for index, argument in enumerate(sys.argv):
        if argument == "--gpu" and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return
        if argument.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = argument.split("=", 1)[1]
            return


configure_visible_gpus()

import torch
from torch.utils.data import DataLoader

from utils.config import (
    CAMUS_SPLIT,
    CAMUS_TASKS,
    DEFAULT_CAMUS_DATA_PATH,
    DEFAULT_ECHONET_DATA_PATH,
    TASKS,
    FinalConfig,
    resolve_data_settings,
)
from utils.data import CardiacVideoDataset
from utils.engine import evaluate, seed_everything, train_one_epoch, unwrap
from models import model_dict


def make_run_name(config: FinalConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_split = f"split-{CAMUS_SPLIT}" if config.task in CAMUS_TASKS else "split-official"
    return f"{timestamp}_{config.task}_{data_split}_loss-{config.mask_loss}"


def arguments():
    parser = argparse.ArgumentParser(description="Train the final CAMUS/EchoNet E3SAM2 model")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=TASKS)
    parser.add_argument("--data", type=Path, help="Optional override for the task-selected data root")
    parser.add_argument("--camus_data_path", "--camus-data-path", type=Path, default=DEFAULT_CAMUS_DATA_PATH)
    parser.add_argument("--echo_data_path", "--echo-data-path", type=Path, default=DEFAULT_ECHONET_DATA_PATH)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("pretrained_checkpoints/sam2_hiera_small.pt"),
        help="SAM2 Hiera-S checkpoint (default: pretrained_checkpoints/sam2_hiera_small.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs"),
        help="Parent directory for timestamped task-specific training runs (default: runs)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--mask-loss", choices=("bce_dice", "structure"), default="bce_dice",
        help="Mask loss used for both training and validation",
    )
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
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
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.workers,
        mask_loss=args.mask_loss,
    )
    run_name = make_run_name(config)
    config.output_dir = args.output / run_name
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_name": run_name,
        "arguments": vars(args),
        "config": asdict(config),
    }
    (config.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    seed_everything(config.seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("The final model must be trained in a CUDA environment")

    print(f"Task: {config.task}, image size: {config.image_size}")
    if config.task in CAMUS_TASKS:
        print(f"CAMUS split: {CAMUS_SPLIT}")
    print(f"Data path: {config.data_path}")
    print(f"Run directory: {config.output_dir}")
    train_set = CardiacVideoDataset(
        config.data_path, config.train_split, config.image_size, config.frame_length,
        training=True,
    )
    val_set = CardiacVideoDataset(
        config.data_path, config.val_split, config.image_size, config.frame_length,
        training=False,
    )
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    model = model_dict.build_model(config, str(args.pretrained)).to(device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.base_lr / config.warmup_iterations,
        betas=(0.9, 0.999), weight_decay=config.weight_decay,
    )
    iteration = 0
    max_iterations = config.epochs * len(train_loader)
    best_dice = -1.0
    log_file = config.output_dir / "metrics.jsonl"
    for epoch in range(config.epochs):
        iteration, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, config, epoch, iteration, max_iterations
        )
        val_metrics = evaluate(model, val_loader, device, config)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(record, ensure_ascii=False))
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(unwrap(model).state_dict(), config.output_dir / "best.pth", _use_new_zipfile_serialization=False)
    torch.save(unwrap(model).state_dict(), config.output_dir / "last.pth", _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    main()
