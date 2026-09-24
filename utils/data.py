"""CAMUS and EchoNet processed-video datasets used by the final experiment."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def load_case(video_path: Path, annotation_path: Path, frame_length: int = 10):
    video = np.load(video_path, allow_pickle=True).swapaxes(0, 1)
    annotation = np.load(annotation_path, allow_pickle=True)
    mask_map = annotation["fnum_mask"].tolist()
    edge_map = annotation["fnum_edge"].tolist()
    frame_numbers = sorted(int(key) for key in mask_map)
    first, last = frame_numbers[0], frame_numbers[-1]
    step = min(last, (last - first) / (frame_length - 1)) if frame_length > 1 else 0.0
    selected = [int(first + step * i) for i in range(frame_length)]
    frames = np.asarray([video[index] for index in selected])

    def lookup(mapping, index):
        return mapping[str(index)] if str(index) in mapping else mapping[index]

    if len(mask_map) == len(video):
        # CAMUS has dense annotations: sample masks/edges at the same indices as frames.
        masks = np.asarray([lookup(mask_map, index) for index in selected])
        edges = np.asarray([lookup(edge_map, index) for index in selected])
    else:
        # EchoNet has sparse ED/ES annotations: retain both labels as [0, -1].
        masks = np.asarray([lookup(mask_map, index) for index in frame_numbers])
        edges = np.asarray([lookup(edge_map, index) for index in frame_numbers])
    metadata = {
        "ef": float(annotation["ef"]),
        "edv": float(annotation["edv"]),
        "esv": float(annotation["esv"]),
        "spacing": np.asarray(annotation["spacing"], dtype=np.float32),
    }
    return frames, masks, edges, metadata


def _positive_point(mask: np.ndarray, random: bool) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask == 1)
    label = 1
    if coordinates.size == 0:
        coordinates = np.argwhere(mask != 1)
        label = 0
    coordinates = coordinates[:, [1, 0]]
    index = np.random.randint(len(coordinates)) if random else len(coordinates) // 2
    return coordinates[index : index + 1].astype(np.float32), np.asarray([label], dtype=np.int64)


class VideoTransform:
    """Video augmentation that preserves input pixels as 0-255 float32."""

    def __init__(self, image_size: int = 256, training: bool = False):
        self.image_size = image_size
        self.training = training

    def __call__(self, frames: np.ndarray, masks: np.ndarray, edges: np.ndarray):
        probability = 0.5 if self.training else 0.0
        frames = frames.astype(np.uint8)
        if np.random.rand() < probability:
            gamma = np.random.randint(10, 25) / 10.0
            frames = (np.power(frames / 255, 1.0 / gamma) * 255).astype(np.uint8)

        frame_images = [TF.to_pil_image(frame.transpose(1, 2, 0)) for frame in frames]
        mask_images = [TF.to_pil_image(mask.astype(np.uint8)) for mask in masks]
        edge_images = [TF.to_pil_image(edge.astype(np.uint8)) for edge in edges]

        # train20 draws from NumPy even for its disabled horizontal-flip branch.
        np.random.rand()
        if np.random.rand() < probability:
            angle = T.RandomRotation.get_params((-30, 30))
            frame_images = [TF.rotate(x, angle) for x in frame_images]
            mask_images = [TF.rotate(x, angle) for x in mask_images]
            edge_images = [TF.rotate(x, angle) for x in edge_images]
        if np.random.rand() < probability:
            scale = np.random.uniform(1, 1.3)
            size = int(self.image_size * scale)
            frame_images = [TF.resize(x, (size, size), InterpolationMode.BILINEAR) for x in frame_images]
            mask_images = [TF.resize(x, (size, size), InterpolationMode.NEAREST) for x in mask_images]
            edge_images = [TF.resize(x, (size, size), InterpolationMode.NEAREST) for x in edge_images]
            top, left, height, width = T.RandomCrop.get_params(
                frame_images[0], (self.image_size, self.image_size)
            )
            frame_images = [TF.crop(x, top, left, height, width) for x in frame_images]
            mask_images = [TF.crop(x, top, left, height, width) for x in mask_images]
            edge_images = [TF.crop(x, top, left, height, width) for x in edge_images]
        if np.random.rand() < probability:
            contrast = T.ColorJitter(contrast=(0.8, 2.0))
            frame_images = [contrast(x) for x in frame_images]
        # Keep the original RNG draw for the disabled random-affine branch too.
        np.random.rand()

        frame_images = [TF.resize(x, [self.image_size, self.image_size], InterpolationMode.BILINEAR) for x in frame_images]
        mask_images = [TF.resize(x, [self.image_size, self.image_size], InterpolationMode.NEAREST) for x in mask_images]
        edge_images = [TF.resize(x, [self.image_size, self.image_size], InterpolationMode.NEAREST) for x in edge_images]
        frames = torch.from_numpy(
            np.stack([np.asarray(x) for x in frame_images]).transpose(0, 3, 1, 2).copy()
        ).float()
        masks = torch.from_numpy((np.stack([np.asarray(x) for x in mask_images]) > 0).astype(np.float32))
        edges = torch.from_numpy((np.stack([np.asarray(x) for x in edge_images]) > 0).astype(np.float32))
        return frames, masks, edges


class CardiacVideoDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int = 256,
                 frame_length: int = 10, training: bool = False):
        self.root = Path(root)
        self.split = split
        self.frame_length = frame_length
        self.transform = VideoTransform(image_size, training)
        self.video_dir = self.root / "videos" / split
        self.annotation_dir = self.root / "annotations" / split
        self.files = sorted(self.video_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy videos found in {self.video_dir}")
        self.training = training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        video_path = self.files[index]
        frames, masks, edges, metadata = load_case(
            video_path, self.annotation_dir / f"{video_path.stem}.npz", self.frame_length
        )
        masks = (masks > 0).astype(np.uint8)
        edges = (edges > 0).astype(np.uint8)
        frames, masks, edges = self.transform(frames, masks, edges)
        points, labels = [], []
        for mask in masks.numpy():
            point, label = _positive_point(mask, self.training)
            points.append(point)
            labels.append(label)
        return {
            "image": frames,
            "label": masks,
            "edge": edges,
            "points": torch.from_numpy(np.stack(points)),
            "point_labels": torch.from_numpy(np.stack(labels)),
            "image_name": video_path.name,
            **{key: torch.as_tensor(value) for key, value in metadata.items()},
        }


# Backward-compatible name for existing imports.
CamusVideoDataset = CardiacVideoDataset
