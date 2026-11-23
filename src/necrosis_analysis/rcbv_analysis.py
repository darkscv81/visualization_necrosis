"""Longitudinal change maps, exploratory thresholds, and region summaries."""

import numpy as np
from scipy import ndimage

from .normalization import summarize


def change_maps(
    may: np.ndarray,
    october: np.ndarray,
    valid: np.ndarray,
    *,
    baseline_floor: float = 0.05,
) -> dict[str, np.ndarray]:
    """Calculate signed and unsigned differences on common rCBV support."""
    if not np.isfinite(baseline_floor) or baseline_floor <= 0:
        raise ValueError("baseline_floor must be finite and strictly positive")
    if may.ndim != 3 or may.shape != october.shape or may.shape != valid.shape or valid.dtype != bool:
        raise ValueError("Change maps require 3D rCBV arrays and a boolean mask on the same grid")
    if not np.isfinite(may[valid]).all() or not np.isfinite(october[valid]).all():
        raise ValueError("Valid rCBV support contains nonfinite values")

    dtype = np.result_type(may.dtype, october.dtype, np.float32)
    signed = np.full(may.shape, np.nan, dtype=dtype)
    np.subtract(october, may, out=signed, where=valid, dtype=dtype)
    ratio_support = valid & (may >= baseline_floor)
    percent = np.full(may.shape, np.nan, dtype=dtype)
    np.divide(signed, may, out=percent, where=ratio_support)
    percent *= 100
    log_ratio = np.full(may.shape, np.nan, dtype=dtype)
    log_support = ratio_support & (october > 0)
    log_ratio[log_support] = np.log(october[log_support]) - np.log(may[log_support])
    return {
        "signed": signed,
        "magnitude": np.abs(signed),
        "percent": percent,
        "positive": np.maximum(signed, 0),
        "negative": np.minimum(signed, 0),
        "log_ratio": log_ratio,
    }


def candidates(
    maps: dict[str, np.ndarray],
    valid: np.ndarray,
    roi: np.ndarray,
    *,
    absolute: float | None = None,
    percent: float | None = None,
    robust_sigma: float | None = None,
    min_region_ml: float = 0,
    voxel_ml: float = 1,
) -> np.ndarray:
    """Label increased (+1) and decreased (-1) regions meeting every enabled threshold."""
    if all(value is None for value in (absolute, percent, robust_sigma)):
        raise ValueError("Enable at least one candidate change threshold")
    for name, value in {"absolute": absolute, "percent": percent, "robust_sigma": robust_sigma}.items():
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and strictly positive when enabled")
    if not np.isfinite(min_region_ml) or min_region_ml < 0:
        raise ValueError("Minimum candidate region volume must be finite and nonnegative")
    if not np.isfinite(voxel_ml) or voxel_ml <= 0:
        raise ValueError("Voxel volume must be finite and strictly positive")

    signed = maps["signed"]
    keep = valid & np.isfinite(signed) & (signed != 0)
    if absolute is not None:
        keep &= np.abs(signed) >= absolute
    if percent is not None:
        keep &= np.abs(maps["percent"]) >= percent
    if robust_sigma is not None:
        reference = signed[roi & valid & np.isfinite(signed)]
        if reference.size < 2:
            raise ValueError("Robust change threshold needs at least two finite reference ROI samples")
        center = np.median(reference)
        spread = 1.4826 * np.median(np.abs(reference - center))
        if spread <= 0:
            raise ValueError("Reference ROI change has zero MAD. Use absolute or percent thresholds")
        keep &= np.abs(signed - center) >= robust_sigma * spread

    labels = np.zeros(signed.shape, dtype=np.int8)
    for direction in (-1, 1):
        direction_mask = keep & (direction * signed > 0)
        if min_region_ml > 0:
            components, _ = ndimage.label(direction_mask)
            counts = np.bincount(components.ravel())
            volumes = counts * voxel_ml
            include = (volumes >= min_region_ml) | np.isclose(volumes, min_region_ml, atol=0, rtol=1e-12)
            include[0] = False
            direction_mask = include[components]
        labels[direction_mask] = direction
    return labels


def candidate_regions(
    candidate: np.ndarray,
    maps: dict[str, np.ndarray],
    *,
    voxel_ml: float,
    dose: np.ndarray | None = None,
    dose_coverage: np.ndarray | None = None,
) -> list[dict]:
    """Describe each 6-connected increased or decreased region."""
    rows = []
    for direction in (-1, 1):
        components, _ = ndimage.label(candidate == direction)
        for component_number, bounds in enumerate(ndimage.find_objects(components), start=1):
            region = components[bounds] == component_number
            location = ndimage.center_of_mass(region)
            centroid = [float(position + extent.start) for position, extent in zip(location, bounds)]
            count = int(np.count_nonzero(region))
            row = {
                "region": len(rows) + 1,
                "direction": "increase" if direction > 0 else "decrease",
                "voxels": count,
                "volume_ml": count * voxel_ml,
                "centroid_z_y_x": centroid,
                "median_signed_change": summarize(maps["signed"][bounds][region])["median"],
                "median_percent_change": summarize(maps["percent"][bounds][region])["median"],
            }
            if dose is not None:
                covered = region if dose_coverage is None else region & dose_coverage[bounds]
                stats = summarize(dose[bounds][covered])
                row.update(dose_evaluable_voxels=stats["n"], mean_dose=stats["mean"], median_dose=stats["median"])
            rows.append(row)
    return rows


def threshold_sensitivity(
    maps: dict[str, np.ndarray],
    valid: np.ndarray,
    roi: np.ndarray,
    *,
    absolute_thresholds: list[float],
    voxel_ml: float,
    percent: float | None = None,
    robust_sigma: float | None = None,
    min_region_ml: float = 0,
    dose: np.ndarray | None = None,
    dose_coverage: np.ndarray | None = None,
    dose_threshold: float | None = None,
) -> list[dict]:
    """Sweep absolute thresholds with the other candidate settings held fixed."""
    rows = []
    for threshold in absolute_thresholds:
        labels = candidates(
            maps, valid, roi, absolute=threshold, percent=percent,
            robust_sigma=robust_sigma, min_region_ml=min_region_ml, voxel_ml=voxel_ml,
        )
        selected = labels != 0
        row = {
            "absolute_threshold": float(threshold),
            "percent_threshold": percent,
            "robust_sigma_multiplier": robust_sigma,
            "increase_ml": int(np.count_nonzero(labels > 0)) * voxel_ml,
            "decrease_ml": int(np.count_nonzero(labels < 0)) * voxel_ml,
            "candidate_ml": int(np.count_nonzero(selected)) * voxel_ml,
        }
        if dose is not None:
            evaluated = selected & np.isfinite(dose)
            if dose_coverage is not None:
                evaluated &= dose_coverage
            stats = summarize(dose[evaluated])
            row.update(
                dose_evaluable_candidate_ml=stats["n"] * voxel_ml,
                mean_dose=stats["mean"],
                median_dose=stats["median"],
                dose_threshold=dose_threshold,
                high_dose_overlap_ml=(
                    int(np.count_nonzero(dose[evaluated] >= dose_threshold)) * voxel_ml
                    if dose_threshold is not None else None
                ),
            )
        rows.append(row)
    return rows
