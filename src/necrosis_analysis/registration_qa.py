"""Independent quantitative checks for a fixed-to-moving LPS registration."""

from collections.abc import Mapping
from itertools import product

import numpy as np
import SimpleITK as sitk

from .image_geometry import (
    compose_transforms,
    index_to_lps,
    resample_mask,
    resample_scalar,
    resample_validity,
    same_geometry,
)


def _entropy(frequencies: np.ndarray) -> float:
    probability = frequencies[frequencies > 0].astype(float)
    probability /= probability.sum()
    return float(-np.sum(probability * np.log(probability)))


def _image_similarity(fixed: np.ndarray, moving: np.ndarray) -> dict:
    r = None
    nmi = None
    if len(fixed) > 1 and np.ptp(fixed) > 0 and np.ptp(moving) > 0:
        # Entropy-normalized MI ranges from 1 for independent images to 2 for
        # identical binned information. Both metrics use the current overlap.
        r = float(np.corrcoef(fixed, moving)[0, 1])
        joint, _, _ = np.histogram2d(fixed, moving, bins=48)
        nmi = (_entropy(joint.sum(axis=0)) + _entropy(joint.sum(axis=1))) / _entropy(joint)
    return {"pearson_r": r, "normalized_mutual_information": nmi}


def _binary_mask(mask: sitk.Image | None, image: sitk.Image) -> sitk.Image | None:
    if mask is None:
        return None
    if not same_geometry(mask, image):
        raise ValueError("QA mask must occupy its corresponding anatomical image grid")
    values = sitk.GetArrayViewFromImage(mask)
    if not np.isin(values, (0, 1)).all() or not values.any():
        raise ValueError("QA masks must be nonempty binary images")
    binary = sitk.Cast(mask, sitk.sitkUInt8)
    # NIfTI stores some geometry fields in single precision. Use the verified
    # anatomical grid exactly for filters with stricter ITK tolerances.
    binary.CopyInformation(image)
    return binary


def _mask_overlap(fixed: sitk.Image, moving_on_fixed: sitk.Image) -> dict:
    fixed_values = sitk.GetArrayViewFromImage(fixed) != 0
    moving_values = sitk.GetArrayViewFromImage(moving_on_fixed) != 0
    dice = 2 * np.count_nonzero(fixed_values & moving_values)
    dice /= np.count_nonzero(fixed_values) + np.count_nonzero(moving_values)
    hausdorff_mm = None
    if moving_values.any():
        distance = sitk.HausdorffDistanceImageFilter()
        distance.Execute(fixed, moving_on_fixed)
        hausdorff_mm = float(distance.GetHausdorffDistance())
    return {"dice": float(dice), "hausdorff_mm": hausdorff_mm}


def _parse_landmarks(landmarks: Mapping | None) -> tuple[np.ndarray, np.ndarray] | None:
    if landmarks is None:
        return None
    try:
        fixed = np.asarray(landmarks["fixed_lps"], dtype=float)
        moving = np.asarray(landmarks["moving_lps"], dtype=float)
    except KeyError as error:
        raise ValueError("Landmarks must include fixed_lps and moving_lps") from error
    if fixed.ndim != 2 or fixed.shape[1] != 3 or fixed.shape != moving.shape or len(fixed) == 0:
        raise ValueError("Landmarks require the same number of (x, y, z) LPS points in both images")
    if not np.isfinite(fixed).all() or not np.isfinite(moving).all():
        raise ValueError("Landmark coordinates must be finite")
    return fixed, moving


def _landmark_errors(
    transform: sitk.Transform, landmarks: tuple[np.ndarray, np.ndarray]
) -> dict:
    fixed, moving = landmarks
    predicted = np.asarray([transform.TransformPoint(tuple(point)) for point in fixed])
    residuals = predicted - moving
    distances = np.linalg.norm(residuals, axis=1)
    return {
        "mean_tre_mm": float(distances.mean()),
        "max_tre_mm": float(distances.max()),
        "landmark_residuals_mm_lps": residuals.tolist(),
        "landmark_distances_mm": distances.tolist(),
    }


def _displacement(fixed: sitk.Image, transform: sitk.Transform) -> dict:
    extent = np.asarray(fixed.GetSize()) - 1
    center = np.array(index_to_lps(fixed, extent / 2))
    center_transformed = np.asarray(transform.TransformPoint(tuple(center)))
    matrix = np.asarray([
        np.asarray(transform.TransformPoint(tuple(center + vector))) - center_transformed
        for vector in np.eye(3)
    ]).T
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, rtol=0, atol=1e-6
    ):
        raise ValueError("Registration QA expects a rigid, proper rotation")
    angle = np.degrees(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2, -1.0, 1.0)))
    corner_points = [index_to_lps(fixed, corner) for corner in product(*zip((0, 0, 0), extent))]
    corner_displacements = [
        np.linalg.norm(np.asarray(transform.TransformPoint(point)) - point)
        for point in corner_points
    ]
    return {
        "center_displacement_mm_lps": (center_transformed - center).tolist(),
        "translation_mm": float(np.linalg.norm(center_transformed - center)),
        "rotation_degrees": float(angle),
        "max_corner_displacement_mm": float(max(corner_displacements)),
    }


def _measure(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    fixed_mask: sitk.Image | None,
    moving_mask: sitk.Image | None,
    landmarks: tuple[np.ndarray, np.ndarray] | None,
) -> dict:
    moving_on_fixed, coverage = resample_scalar(moving, fixed, transform)
    reciprocal_overlap = resample_validity(fixed, moving, transform.GetInverse())
    fixed_values = sitk.GetArrayViewFromImage(fixed)
    moving_values = sitk.GetArrayViewFromImage(moving)
    fixed_valid = np.isfinite(fixed_values)
    moving_valid = np.isfinite(moving_values)
    overlap = fixed_valid & (sitk.GetArrayViewFromImage(coverage) != 0)
    reciprocal_overlap &= moving_valid
    metrics = {
        "fixed_coverage_fraction": float(overlap.sum() / fixed_valid.sum()) if fixed_valid.any() else 0.0,
        "moving_coverage_fraction": (
            float(reciprocal_overlap.sum() / moving_valid.sum()) if moving_valid.any() else 0.0
        ),
        "fixed_overlap_voxels": int(overlap.sum()),
    }
    similarity_mask = overlap.copy()
    if fixed_mask is not None:
        similarity_mask &= sitk.GetArrayViewFromImage(fixed_mask) != 0
    if moving_mask is not None:
        moving_on_fixed_mask = resample_mask(moving_mask, fixed, transform)
        similarity_mask &= sitk.GetArrayViewFromImage(moving_on_fixed_mask) != 0
        if fixed_mask is not None:
            metrics.update(_mask_overlap(fixed_mask, moving_on_fixed_mask))
    metrics.update(_image_similarity(
        fixed_values[similarity_mask], sitk.GetArrayViewFromImage(moving_on_fixed)[similarity_mask]
    ))
    if landmarks is not None:
        metrics.update(_landmark_errors(transform, landmarks))
    return metrics


def evaluate_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    *,
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    landmarks: Mapping | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict:
    """Measure coverage and alignment before and after applying the estimated transform.

    Independent binary masks on native image grids enable Dice and Hausdorff
    measurements on the fixed grid. Landmarks use corresponding fixed_lps and
    moving_lps arrays in physical millimeters. Threshold flags supplement visual
    review and manual sign-off.
    """
    fixed_mask = _binary_mask(fixed_mask, fixed)
    moving_mask = _binary_mask(moving_mask, moving)
    points = _parse_landmarks(landmarks)
    displacement = _displacement(fixed, transform)
    before = _measure(fixed, moving, compose_transforms(), fixed_mask, moving_mask, points)
    after = _measure(fixed, moving, transform, fixed_mask, moving_mask, points)
    measures = {**after, **displacement}
    threshold_metrics = {
        "min_fixed_coverage": "fixed_coverage_fraction",
        "min_moving_coverage": "moving_coverage_fraction",
        "min_dice": "dice",
        "max_hausdorff_mm": "hausdorff_mm",
        "max_mean_tre_mm": "mean_tre_mm",
        "max_tre_mm": "max_tre_mm",
        "max_translation_mm": "translation_mm",
        "max_rotation_degrees": "rotation_degrees",
    }
    warnings = []
    if after["fixed_overlap_voxels"] == 0:
        warnings.append("Registered anatomy has no finite overlap")
    if after["normalized_mutual_information"] is None:
        warnings.append("Registered overlap has insufficient samples or intensity variation for similarity QA")
    for threshold_key, limit in (thresholds or {}).items():
        if threshold_key not in threshold_metrics:
            raise ValueError(f"Unknown registration QA threshold {threshold_key!r}")
        if not np.isfinite(limit) or limit < 0:
            raise ValueError(f"Registration QA threshold {threshold_key!r} must be finite and nonnegative")
        metric = threshold_metrics[threshold_key]
        value = measures.get(metric)
        if value is None:
            warnings.append(f"{metric} unavailable for configured threshold {threshold_key}")
        elif (threshold_key.startswith("min_") and value < limit) or (
            threshold_key.startswith("max_") and value > limit
        ):
            warnings.append(f"{metric} of {value:.4g} violates {threshold_key} of {limit:.4g}")
    return {
        "direction": "fixed LPS to moving LPS",
        "before": before,
        "after": after,
        "displacement": displacement,
        "warnings": warnings,
    }
