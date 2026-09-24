"""Prepare EchoNet-Dynamic data with Sobel-dilation edge targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional

try:
    from .utils_contour import echonet_trace_to_mask
except ImportError:  # Support direct execution: python utils/preprocess_echonet_edge.py
    from utils_contour import echonet_trace_to_mask


RESIZED_SIZE = 128
SPLIT_NAMES = ("train", "val", "test")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _add_common_arguments(parser: argparse.ArgumentParser, default_output: str) -> None:
    parser.add_argument(
        "-i",
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        default="data/EchoNet-Dynamic",
        help="EchoNet-Dynamic root directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default=default_output,
        help="Destination directory.",
    )
    parser.add_argument(
        "-kpts",
        "--save_kpts",
        "--save-kpts",
        dest="save_kpts",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Save per-frame NPZ annotations (default: true).",
    )
    parser.add_argument(
        "-masks",
        "--save_masks",
        "--save-masks",
        dest="save_masks",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Save per-frame mask PNGs (default: true).",
    )
    parser.add_argument(
        "-imgs",
        "--save_imgs",
        "--save-imgs",
        dest="save_imgs",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Save annotated video frames as PNGs (default: true).",
    )


def loadvideo(filename: str | Path) -> np.ndarray:
    """Load an RGB video as a uint8 array with shape [C, T, H, W]."""
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Invalid video metadata: {path}")

        video = np.empty((frame_count, frame_height, frame_width, 3), dtype=np.uint8)
        for frame_index in range(frame_count):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Failed to load frame {frame_index} from {path}")
            video[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    return video.transpose(3, 0, 1, 2)


def _resize_video(video: np.ndarray, resized_size: int) -> np.ndarray:
    video_tchw = torch.from_numpy(video).permute(1, 0, 2, 3)
    resized = tv_functional.resize(
        video_tchw,
        (resized_size, resized_size),
        interpolation=InterpolationMode.BILINEAR,
    )
    return resized.numpy().transpose(1, 0, 2, 3)


def _prepare_output_dirs(output_root: Path) -> None:
    for split_name in SPLIT_NAMES:
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "annotations" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "echocycle" / "videos" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "echocycle" / "annotations" / split_name).mkdir(
            parents=True,
            exist_ok=True,
        )


def _trace_points(trace: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = trace[1:, 0], trace[1:, 1], trace[1:, 2], trace[1:, 3]
    x = np.concatenate((x1, np.flip(x2)))
    y = np.concatenate((y1, np.flip(y2)))
    return np.column_stack((x, y))


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _write_filename_lists(
    output_root: Path,
    frame_names: dict[str, list[str]],
    cycle_names: dict[str, list[str]],
    invalid_names: list[str],
) -> None:
    for split_name in SPLIT_NAMES:
        _write_lines(
            output_root / f"echonet_{split_name}_filenames.txt",
            frame_names[split_name],
        )
        _write_lines(
            output_root / "echocycle" / f"echonet_cycle_{split_name}_filenames.txt",
            cycle_names[split_name],
        )
    _write_lines(output_root / "echonet_invalid_filenames.txt", invalid_names)


class EdgeGenerator(nn.Module):
    """Generate binary edge targets with Sobel filtering and dilation.

    Sobel filtering first identifies the precise mask boundary. Morphological
    dilation then thickens that boundary so edge supervision is less sensitive
    to small spatial misalignments.
    """

    def __init__(self, dilation_kernel_size: int = 3) -> None:
        super().__init__()
        if dilation_kernel_size <= 0 or dilation_kernel_size % 2 == 0:
            raise ValueError("dilation_kernel_size must be a positive odd integer")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.dilation_kernel_size = dilation_kernel_size
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _as_bchw(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        original_dim = mask.ndim
        if original_dim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            mask = mask.unsqueeze(1)
        elif original_dim != 4:
            raise ValueError(f"Expected a 2D, 3D, or 4D mask tensor, got {mask.shape}")

        if mask.shape[1] != 1:
            raise ValueError(f"Expected a single-channel mask, got {mask.shape}")
        return mask.float(), original_dim

    @staticmethod
    def _restore_shape(edge: torch.Tensor, original_dim: int) -> torch.Tensor:
        if original_dim == 2:
            return edge[0, 0]
        if original_dim == 3:
            return edge[:, 0]
        return edge

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        mask_bchw, original_dim = self._as_bchw(mask)
        mask_bchw = (mask_bchw > 0).float()

        grad_x = F.conv2d(mask_bchw, self.sobel_x, padding=1)
        grad_y = F.conv2d(mask_bchw, self.sobel_y, padding=1)
        sharp_edge = (grad_x.square() + grad_y.square() > 0).float()
        dilated_edge = F.max_pool2d(
            sharp_edge,
            kernel_size=self.dilation_kernel_size,
            stride=1,
            padding=self.dilation_kernel_size // 2,
        )
        return self._restore_shape(dilated_edge, original_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_arguments(
        parser,
        default_output=(
            "data/EchoNet_Processed/"
            "Resized128_Edge_Kernel_Size3"
        ),
    )
    parser.add_argument(
        "-d",
        "--dilation_kernel_size",
        "--dilation-kernel-size",
        dest="dilation_kernel_size",
        type=int,
        default=3,
        help="Positive odd dilation kernel size for Sobel edge targets.",
    )
    return parser.parse_args()


def preprocess_data(
    input_path: str,
    output_path: str,
    save_kpts: bool,
    save_masks: bool,
    save_imgs: bool,
    resized_size: int,
    dilation_kernel_size: int = 3,
) -> None:
    input_root = Path(input_path)
    output_root = Path(output_path)
    tracings_path = input_root / "VolumeTracings.csv"
    file_list_path = input_root / "FileList.csv"
    if not tracings_path.is_file():
        raise ValueError(f"Missing volume tracings: {tracings_path}")
    if not file_list_path.is_file():
        raise ValueError(f"Missing file list: {file_list_path}")

    tracings = pd.read_csv(tracings_path)
    metadata = pd.read_csv(file_list_path)
    _prepare_output_dirs(output_root)

    frame_names = {split_name: [] for split_name in SPLIT_NAMES}
    cycle_names = {split_name: [] for split_name in SPLIT_NAMES}
    invalid_names: list[str] = []
    invalid_set: set[str] = set()
    edge_generator = EdgeGenerator(dilation_kernel_size=dilation_kernel_size)

    def mark_invalid(name: str, reason: str) -> None:
        if name not in invalid_set:
            invalid_set.add(name)
            invalid_names.append(name)
        print(f"Skipping {name}: {reason}")

    for video_index, row in enumerate(metadata.itertuples(index=False), start=1):
        raw_name = str(row.FileName)
        name = Path(raw_name).stem
        filename = raw_name if Path(raw_name).suffix else f"{raw_name}.avi"
        split_name = str(row.Split).strip().lower()
        if split_name not in SPLIT_NAMES:
            mark_invalid(name, f"unknown split {row.Split!r}")
            continue

        video_path = input_root / "Videos" / filename
        if not video_path.is_file():
            mark_invalid(name, f"video not found at {video_path}")
            continue

        patient_traces = tracings[tracings.FileName == filename]
        frame_indices = sorted(int(value) for value in patient_traces.Frame.unique())
        if len(frame_indices) != 2:
            mark_invalid(name, f"expected two annotated frames, found {len(frame_indices)}")
            continue

        video = _resize_video(loadvideo(video_path), resized_size)
        annotations: list[tuple[int, np.ndarray, np.ndarray]] = []
        valid = True

        for frame_index in frame_indices:
            frame_trace = patient_traces[patient_traces.Frame == frame_index]
            trace = np.asarray(frame_trace.loc[:, "X1":"Y2"])
            if trace.size // 2 != 42:
                mark_invalid(name, f"frame {frame_index} has an invalid number of trace points")
                valid = False
                break
            if frame_index < 0 or frame_index >= video.shape[1]:
                mark_invalid(name, f"frame index {frame_index} is outside the video")
                valid = False
                break

            mask = echonet_trace_to_mask(trace, (112, 112))
            mask = cv2.resize(
                mask,
                (resized_size, resized_size),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0).astype(np.uint8)
            annotations.append((frame_index, mask, _trace_points(trace)))

        if not valid:
            continue

        print(f"Video {video_index}: {filename}")
        images_dir = output_root / "images" / split_name
        annotations_dir = output_root / "annotations" / split_name
        cycle_video_dir = output_root / "echocycle" / "videos" / split_name
        cycle_annotation_dir = output_root / "echocycle" / "annotations" / split_name

        frame_images: dict[str, np.ndarray] = {}
        frame_masks: dict[str, np.ndarray] = {}
        frame_edges: dict[str, np.ndarray] = {}
        keypoints: list[np.ndarray] = []

        with torch.no_grad():
            for frame_index, mask, points in annotations:
                key = str(frame_index)
                frame_image = video[:, frame_index].transpose(1, 2, 0)
                frame_images[key] = frame_image
                frame_masks[key] = mask
                keypoints.append(points)

                edge = edge_generator(torch.from_numpy(mask))
                frame_edges[key] = edge.detach().cpu().numpy().astype(np.uint8)

                frame_stem = f"{name}_{frame_index}"
                frame_names[split_name].append(f"{frame_stem}.png")
                if save_imgs:
                    imageio.imwrite(images_dir / f"{frame_stem}.png", frame_image)
                if save_masks:
                    imageio.imwrite(annotations_dir / f"{frame_stem}.png", mask)
                if save_kpts:
                    np.savez(
                        annotations_dir / f"{frame_stem}.npz",
                        mask=mask,
                        kpts=points,
                        ef=float(row.EF),
                    )

        np.save(cycle_video_dir / f"{name}.npy", video)
        np.savez(
            cycle_annotation_dir / f"{name}.npz",
            fnum=frame_images,
            fnum_mask=frame_masks,
            fnum_edge=frame_edges,
            kpts=np.asarray(keypoints),
            ef=float(row.EF),
            edv=float(row.EDV),
            esv=float(row.ESV),
            spacing=(1.0, 1.0, 1.0),
        )
        cycle_names[split_name].append(f"{name}.npy")

    _write_filename_lists(output_root, frame_names, cycle_names, invalid_names)


def main() -> None:
    args = parse_args()
    if not Path(args.input_dir).is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    preprocess_data(
        args.input_dir,
        args.output_dir,
        args.save_kpts,
        args.save_masks,
        args.save_imgs,
        RESIZED_SIZE,
        args.dilation_kernel_size,
    )


if __name__ == "__main__":
    main()
