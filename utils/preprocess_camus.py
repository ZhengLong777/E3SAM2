"""Prepare CAMUS videos and LV annotations without edge targets.

Outputs:
    videos/{split}/{patient}_{view}.npy
    annotations/{split}/{patient}_{view}.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

try:
    from .compute_ef import compute_left_ventricle_volumes
except ImportError:  # Support direct execution: python utils/preprocess_camus.py
    from compute_ef import compute_left_ventricle_volumes


RESIZE_SIZE = (256, 256)
SPLIT_RATIOS = (0.7, 0.1, 0.2)
PATIENT_COUNT = 500
VIEWS = ("2CH", "4CH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        default="data/CAMUS_Source/database_nifti",
        help="CAMUS database_nifti directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="data/CAMUS_Processed/NoEdge",
        help="Destination directory.",
    )
    parser.add_argument(
        "-f",
        "--split_file",
        "--split-file",
        dest="split_file",
        help="Optional JSON file with train, val, and test patient lists.",
    )
    return parser.parse_args()


def _resample_xy(image: sitk.Image, size: tuple[int, int]) -> sitk.Image:
    """Resize the first two axes while preserving physical image extent."""
    old_x, old_y, old_z = image.GetSize()
    spacing_x, spacing_y, spacing_z = image.GetSpacing()
    new_x, new_y = size

    new_spacing = (
        spacing_x * old_x / new_x,
        spacing_y * old_y / new_y,
        spacing_z,
    )
    return sitk.Resample(
        image,
        (new_x, new_y, old_z),
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
    )


def _load_sitk(
    filepath: str | Path,
    resize_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = sitk.ReadImage(str(filepath))
    if resize_size is not None:
        image = _resample_xy(image, resize_size)

    metadata = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
    }
    return np.squeeze(sitk.GetArrayFromImage(image)), metadata


def _read_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, value = line.split(":", maxsplit=1)
            values[key.strip()] = value.strip()
    return values


def _patient_names() -> list[str]:
    return [f"patient{index:04d}" for index in range(1, PATIENT_COUNT + 1)]


def _load_splits(
    patient_names: list[str],
    split_file: str | Path | None,
) -> dict[str, list[str]]:
    if split_file:
        with Path(split_file).open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        return {name: list(split_data[name]) for name in ("train", "val", "test")}

    train_end = int(SPLIT_RATIOS[0] * len(patient_names))
    val_end = train_end + int(SPLIT_RATIOS[1] * len(patient_names))
    return {
        "train": patient_names[:train_end],
        "val": patient_names[train_end:val_end],
        "test": patient_names[val_end:],
    }


def _compute_patient_ef(patient_dir: Path) -> tuple[float, float, float]:
    pattern = "{patient}_{view}_{instant}_gt.nii.gz"
    masks: dict[tuple[str, str], np.ndarray] = {}
    spacings: dict[str, tuple[float, float]] = {}

    for view in VIEWS:
        for instant in ("ED", "ES"):
            path = patient_dir / pattern.format(
                patient=patient_dir.name,
                view=view,
                instant=instant,
            )
            mask, metadata = _load_sitk(path)
            masks[(view, instant)] = mask == 1
            if instant == "ED":
                spacings[view] = metadata["spacing"][:2][::-1]

    edv, esv = compute_left_ventricle_volumes(
        masks[("2CH", "ED")],
        masks[("2CH", "ES")],
        spacings["2CH"],
        masks[("4CH", "ED")],
        masks[("4CH", "ES")],
        spacings["4CH"],
    )
    if edv <= 0:
        raise ValueError(f"Non-positive EDV for {patient_dir.name}: {edv}")
    ef = round(100.0 * (edv - esv) / edv, 2)
    return ef, edv, esv


def _prepare_output_dirs(output_path: Path) -> None:
    for category in ("videos", "annotations"):
        for split_name in ("train", "val", "test"):
            (output_path / category / split_name).mkdir(parents=True, exist_ok=True)


def _write_split_files(output_path: Path, splits: dict[str, list[str]]) -> None:
    for split_name, patient_names in splits.items():
        split_path = output_path / f"camus_{split_name}_filenames.txt"
        split_path.write_text("".join(f"{name}\n" for name in patient_names), encoding="utf-8")


def preprocess_data(input_path: str, output_path: str, split_file: str | None) -> None:
    input_root = Path(input_path)
    output_root = Path(output_path)
    patient_names = _patient_names()
    splits = _load_splits(patient_names, split_file)
    split_by_patient = {
        patient: split_name
        for split_name, patients in splits.items()
        for patient in patients
    }

    _prepare_output_dirs(output_root)

    for patient_index, patient_name in enumerate(patient_names, start=1):
        split_name = split_by_patient.get(patient_name)
        if split_name is None:
            continue

        patient_dir = input_root / patient_name
        ef, edv, esv = _compute_patient_ef(patient_dir)

        for view in VIEWS:
            sequence_path = patient_dir / f"{patient_name}_{view}_half_sequence.nii.gz"
            mask_path = patient_dir / f"{patient_name}_{view}_half_sequence_gt.nii.gz"
            sequence, sequence_info = _load_sitk(sequence_path, RESIZE_SIZE)
            masks, mask_info = _load_sitk(mask_path, RESIZE_SIZE)

            if sequence_info["spacing"] != mask_info["spacing"]:
                raise ValueError(f"Image/mask spacing mismatch for {patient_name} {view}")

            cfg = _read_cfg(patient_dir / f"Info_{view}.cfg")
            if cfg["ED"] != cfg["NbFrame"] and cfg["ES"] != cfg["NbFrame"]:
                raise ValueError(f"Unexpected cardiac-cycle endpoints for {patient_name} {view}")

            if int(cfg["ED"]) > int(cfg["ES"]):
                sequence = np.flip(sequence, axis=0)
                masks = np.flip(masks, axis=0)

            video = np.repeat(sequence[np.newaxis, ...], 3, axis=0)
            frame_masks = {
                str(frame_index): (frame_mask == 1).astype(np.uint8)
                for frame_index, frame_mask in enumerate(masks)
            }

            stem = f"{patient_name}_{view}"
            np.save(output_root / "videos" / split_name / f"{stem}.npy", video)
            np.savez(
                output_root / "annotations" / split_name / f"{stem}.npz",
                fnum_mask=frame_masks,
                ef=ef,
                edv=edv,
                esv=esv,
                spacing=sequence_info["spacing"],
            )
            print(patient_index, f"{stem}.npy")

    _write_split_files(output_root, splits)


def main() -> None:
    args = parse_args()
    if not Path(args.input_dir).is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    preprocess_data(args.input_dir, args.output_dir, args.split_file)


if __name__ == "__main__":
    main()
