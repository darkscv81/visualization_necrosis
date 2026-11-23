"""Longitudinal perfusion normalization and descriptive distribution summaries."""

import numpy as np


def summarize(values: np.ndarray) -> dict:
    """Summarize finite values using population standard deviation."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict.fromkeys(
            ("mean", "median", "std", "iqr", "p05", "p25", "p75", "p95", "min", "max")
        ) | {"n": 0}
    p05, p25, median, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "median": float(median),
        "std": float(np.std(values, dtype=np.float64)),
        "iqr": float(p75 - p25),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def normalize(
    may: np.ndarray,
    october: np.ndarray,
    valid: np.ndarray,
    roi: np.ndarray,
    *,
    method: str = "white_matter",
    min_roi_voxels: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply per-scan medians, one shared May divisor, or the existing scale."""
    if may.ndim != 3 or any(array.shape != may.shape for array in (october, valid, roi)):
        raise ValueError("Normalization requires 3D rCBV images and masks on the same grid")
    if valid.dtype != bool or roi.dtype != bool or np.any(roi & ~valid):
        raise ValueError("The reference ROI must be a boolean subset of valid common coverage")
    if not np.isfinite(may[valid]).all() or not np.isfinite(october[valid]).all():
        raise ValueError("Valid rCBV coverage contains nonfinite values")
    if not isinstance(min_roi_voxels, (int, np.integer)) or min_roi_voxels < 1:
        raise ValueError("min_roi_voxels must be a positive integer")
    roi_voxels = int(np.count_nonzero(roi))
    if roi_voxels < min_roi_voxels:
        raise ValueError(
            f"White-matter ROI has {roi_voxels} eligible voxels, fewer than the required {min_roi_voxels}"
        )
    raw_reference_may = summarize(may[roi])
    raw_reference_october = summarize(october[roi])
    median_may = raw_reference_may["median"]
    median_october = raw_reference_october["median"]
    if min(median_may, median_october) <= 0:
        raise ValueError("Both white-matter rCBV medians must be positive. Review the ROI and source scaling")

    strategies = {
        "white_matter": (median_may, median_october),
        "shared_may": (median_may, median_may),
        "already_normalized": (1.0, 1.0),
    }
    if method not in strategies:
        raise ValueError(f"Unknown normalization method {method!r}. Choose from {', '.join(strategies)}")

    divisor_may, divisor_october = strategies[method]
    dtype = np.result_type(may.dtype, october.dtype, np.float32)
    normalized_may = np.full(may.shape, np.nan, dtype=dtype)
    normalized_october = np.full(may.shape, np.nan, dtype=dtype)
    np.divide(may, divisor_may, out=normalized_may, where=valid)
    np.divide(october, divisor_october, out=normalized_october, where=valid)
    if not np.isfinite(normalized_may[valid]).all() or not np.isfinite(normalized_october[valid]).all():
        raise ValueError("Nonfinite normalized rCBV in valid coverage. Check the reference medians and scale")
    raw_may = summarize(may[valid])
    raw_october = summarize(october[valid])
    return normalized_may, normalized_october, {
        "method": method,
        "factors": {"may": divisor_may, "october": divisor_october},
        "roi_voxels": roi_voxels,
        "raw": {"may": raw_may, "october": raw_october},
        "raw_reference": {"may": raw_reference_may, "october": raw_reference_october},
        "normalized": {
            "may": summarize(normalized_may[valid]),
            "october": summarize(normalized_october[valid]),
        },
        "normalized_reference": {
            "may": summarize(normalized_may[roi]),
            "october": summarize(normalized_october[roi]),
        },
        "negative_raw_voxels": {
            "may": int(np.count_nonzero(may[valid] < 0)),
            "october": int(np.count_nonzero(october[valid] < 0)),
        },
        "strategy_comparison": {
            strategy: {
                "may_divisor": may_factor,
                "october_divisor": october_factor,
                "reference_median_change": median_october / october_factor - median_may / may_factor,
                "mean_common_brain_change": raw_october["mean"] / october_factor - raw_may["mean"] / may_factor,
            }
            for strategy, (may_factor, october_factor) in strategies.items()
        },
    }
