"""
EF utilities used by CAMUS and EchoNet evaluation.

CAMUS uses biplane Simpson's method from 2CH and 4CH masks with light mask
cleanup for stable contour fitting. EchoNet uses the PCA-based single-plane
approximation.
"""

import logging
from typing import Tuple

import numpy as np
import PIL
from PIL.Image import Resampling
from skimage.measure import find_contours, label, regionprops

try:
    from scipy.ndimage import binary_fill_holes
except ImportError:  # pragma: no cover - optional robustness
    binary_fill_holes = None


logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, size: Tuple[int, int],
                 resample: Resampling = Resampling.NEAREST) -> np.ndarray:
    return np.array(PIL.Image.fromarray(image).resize(size, resample=resample))


def resize_image_to_isotropic(
    image: np.ndarray,
    spacing: Tuple[float, float],
    resample: Resampling = Resampling.NEAREST,
) -> Tuple[np.ndarray, float]:
    spacing = (float(spacing[0]), float(spacing[1]))
    scaling = np.array(spacing) / min(spacing)
    new_height, new_width = (np.array(image.shape) * scaling).round().astype(int)
    return resize_image(image, (new_width, new_height), resample=resample), min(spacing)


def compute_left_ventricle_volumes(
    a2c_ed: np.ndarray,
    a2c_es: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
) -> Tuple[float, float]:
    """Compute CAMUS biplane EDV/ESV in mL from 2CH and 4CH LV masks."""
    for mask_name, mask in [
        ("a2c_ed", a2c_ed), ("a2c_es", a2c_es),
        ("a4c_ed", a4c_ed), ("a4c_es", a4c_es),
    ]:
        if mask.max() > 1:
            logger.warning(
                f"`compute_left_ventricle_volumes` expects binary segmentation masks of the left ventricle (LV). "
                f"However, the `{mask_name}` segmentation contains a label greater than '1/True'. If this was done "
                f"voluntarily, you can safely ignore this warning. However, the most likely cause is that you forgot "
                f"to extract the binary LV segmentation from a multi-class segmentation mask."
            )

    a2c_ed_d, a2c_ed_h = _compute_diameters(a2c_ed, a2c_voxelspacing)
    a2c_es_d, a2c_es_h = _compute_diameters(a2c_es, a2c_voxelspacing)
    a4c_ed_d, a4c_ed_h = _compute_diameters(a4c_ed, a4c_voxelspacing)
    a4c_es_d, a4c_es_h = _compute_diameters(a4c_es, a4c_voxelspacing)
    step_size = max(a2c_ed_h, a2c_es_h, a4c_ed_h, a4c_es_h)

    ed_volume = _compute_left_ventricle_volume_by_instant(a2c_ed_d, a4c_ed_d, step_size)
    es_volume = _compute_left_ventricle_volume_by_instant(a2c_es_d, a4c_es_d, step_size)
    return ed_volume, es_volume


def _compute_left_ventricle_volume_by_instant(
    a2c_diameters: np.ndarray,
    a4c_diameters: np.ndarray,
    step_size: float,
) -> float:
    a2c_diameters = a2c_diameters / 1000.0
    a4c_diameters = a4c_diameters / 1000.0
    step_size = step_size / 1000.0
    lv_volume = np.sum(a2c_diameters * a4c_diameters) * step_size * np.pi / 4.0
    return round(float(lv_volume * 1e6), 2)


def _find_distance_to_edge(
    segmentation: np.ndarray,
    point_on_mid_line: np.ndarray,
    normal_direction: np.ndarray,
) -> float:
    distance = 8.0
    while True:
        current_position = point_on_mid_line + distance * normal_direction
        y, x = np.round(current_position).astype(int)
        if segmentation.shape[0] <= y or y < 0 or segmentation.shape[1] <= x or x < 0:
            return distance
        if segmentation[y, x] == 0:
            return distance
        distance += 0.5


def _distance_line_to_points(
    line_point_0: np.ndarray,
    line_point_1: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    return np.absolute(np.cross(line_point_1 - line_point_0, line_point_0 - points)) / np.linalg.norm(
        line_point_1 - line_point_0
    )


def _get_angle_of_lines_to_point(reference_point: np.ndarray, moving_points: np.ndarray) -> np.ndarray:
    diff = moving_points - reference_point
    return abs(np.degrees(np.arctan2(diff[:, 0], diff[:, 1])))


def _compute_diameters(segmentation: np.ndarray, voxelspacing: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    segmentation = _preprocess_lv_mask(segmentation)
    segmentation, isotropic_spacing = resize_image_to_isotropic(segmentation, voxelspacing)
    segmentation = (segmentation > 0).astype(np.uint8)

    contours = find_contours(segmentation, 0.5)
    if not contours:
        raise ValueError("No LV contour found.")
    contour = max(contours, key=len)
    return _compute_diameters_from_contour(segmentation, contour, isotropic_spacing)


def _compute_diameters_from_contour(
    segmentation: np.ndarray,
    contour: np.ndarray,
    isotropic_spacing: float,
) -> Tuple[np.ndarray, float]:
    best_i, best_j = None, None

    best_length = 0.0
    for point_idx in range(2, len(contour)):
        previous_points = contour[:point_idx]
        angles_to_previous = _get_angle_of_lines_to_point(contour[point_idx], previous_points)
        for acute_angle_idx in np.nonzero(angles_to_previous <= 45)[0]:
            intermediate_points = contour[acute_angle_idx + 1: point_idx]
            distance_to_intermediate = _distance_line_to_points(
                contour[point_idx], contour[acute_angle_idx], intermediate_points
            )
            if np.all(distance_to_intermediate <= 8):
                distance = np.linalg.norm(contour[point_idx] - contour[acute_angle_idx])
                if best_length < distance:
                    best_length = distance
                    best_i = point_idx
                    best_j = acute_angle_idx

    if best_i is None or best_j is None:
        raise ValueError("Could not estimate the mitral valve plane.")

    mid_point = int(best_j + round((best_i - best_j) / 2))
    mid_line_length = 0.0
    apex = 0
    for i in range(len(contour)):
        length = np.linalg.norm(contour[mid_point] - contour[i])
        if mid_line_length < length:
            mid_line_length = length
            apex = i

    direction = contour[apex] - contour[mid_point]
    normal_direction = np.array([-direction[1], direction[0]])
    normal_direction = normal_direction / np.linalg.norm(normal_direction)

    diameters = []
    for fraction in np.linspace(0, 1, 20, endpoint=False):
        point_on_mid_line = contour[mid_point] + direction * fraction
        distance1 = _find_distance_to_edge(segmentation, point_on_mid_line, normal_direction)
        distance2 = _find_distance_to_edge(segmentation, point_on_mid_line, -normal_direction)
        diameters.append((distance1 + distance2) * isotropic_spacing)

    step_size = (mid_line_length * isotropic_spacing) / 20.0
    return np.array(diameters), step_size


def _preprocess_lv_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        raise ValueError("Empty LV mask.")

    lab = label(mask)
    props = regionprops(lab)
    if not props:
        raise ValueError("No connected component found in LV mask.")

    largest = max(props, key=lambda x: x.area)
    mask = (lab == largest.label).astype(np.uint8)
    if binary_fill_holes is not None:
        mask = binary_fill_holes(mask).astype(np.uint8)
    return mask


def compute_left_ventricle_volumes_single_plane(
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float] = (1.0, 1.0),
    n_slices: int = 20,
) -> Tuple[float, float]:
    """Compute single-plane EDV/ESV in mL from ED and ES LV masks."""
    sp = (float(a4c_voxelspacing[0]), float(a4c_voxelspacing[1]))
    ed_d, ed_h = _compute_diameters_single_plane_pca(a4c_ed, sp, n_slices)
    es_d, es_h = _compute_diameters_single_plane_pca(a4c_es, sp, n_slices)
    return (
        _compute_volume_single_plane_simpson(ed_d, ed_h),
        _compute_volume_single_plane_simpson(es_d, es_h),
    )


def compute_ef_single_plane(
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float] = (1.0, 1.0),
    n_slices: int = 20,
) -> float:
    edv, esv = compute_left_ventricle_volumes_single_plane(
        a4c_ed, a4c_es, a4c_voxelspacing, n_slices
    )
    if edv <= 0:
        return 0.0
    return round((edv - esv) / edv * 100.0, 2)


def _compute_diameters_single_plane_pca(
    segmentation: np.ndarray,
    voxelspacing: Tuple[float, float],
    n_slices: int = 20,
) -> Tuple[np.ndarray, float]:
    segmentation = _preprocess_lv_mask(segmentation)
    segmentation, iso_sp = resize_image_to_isotropic(segmentation, voxelspacing)
    segmentation = (segmentation > 0).astype(np.uint8)

    ys, xs = np.where(segmentation > 0)
    if len(ys) < 10:
        return np.zeros(n_slices), 0.0

    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, -1]
    minor_axis = eigvecs[:, 0]

    proj_major = centered @ major_axis
    proj_minor = centered @ minor_axis
    p_min, p_max = proj_major.min(), proj_major.max()
    long_len_px = p_max - p_min
    if long_len_px < 2:
        return np.zeros(n_slices), 0.0

    bin_edges = np.linspace(p_min, p_max, n_slices + 1)
    diameters_mm = np.zeros(n_slices)
    for i in range(n_slices):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (proj_major >= lo) & (proj_major < hi) if i < n_slices - 1 else (
            (proj_major >= lo) & (proj_major <= hi)
        )
        if in_bin.sum() > 0:
            minor_vals = proj_minor[in_bin]
            diameters_mm[i] = (minor_vals.max() - minor_vals.min()) * iso_sp

    step_size_mm = (long_len_px * iso_sp) / n_slices
    return diameters_mm, step_size_mm


def _compute_volume_single_plane_simpson(diameters_mm: np.ndarray, step_size_mm: float) -> float:
    d_m = diameters_mm / 1000.0
    h_m = step_size_mm / 1000.0
    vol_m3 = (np.pi / 4.0) * np.sum(d_m ** 2) * h_m
    return round(float(vol_m3 * 1e6), 2)
