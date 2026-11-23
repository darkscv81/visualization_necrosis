"""Load reviewed masks and define the common-perfusion analysis support."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_mask(path: str | Path, reference: sitk.Image) -> np.ndarray:
    """Resample an exported label mask in the reviewed May LPS frame.

    The caller must establish that the mask and reference share a patient frame.
    All nonzero labels are included. Nearest-neighbor interpolation preserves labels.
    """
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"Expected a scalar 3D mask at {path}")
    values = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"Mask at {path} must contain finite nonnegative integer labels")
    binary = sitk.Cast(image > 0, sitk.sitkUInt8)
    aligned = sitk.Resample(binary, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor)
    mask = sitk.GetArrayFromImage(aligned).astype(bool)
    if np.any(values) and not mask.any():
        raise ValueError(f"Nonempty mask at {path} has no sampled voxels on the May grid")
    return mask


def analysis_masks(
    brain: np.ndarray,
    reference_roi: np.ndarray,
    may: np.ndarray,
    october: np.ndarray,
    *,
    exclusions: np.ndarray | None = None,
    may_coverage: np.ndarray | None = None,
    october_coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect the reviewed brain and reference masks with common finite coverage."""
    if may.ndim != 3 or may.shape != october.shape:
        raise ValueError("May and October rCBV must be 3D arrays on the same grid")
    supplied_masks = {
        "brain": brain,
        "reference ROI": reference_roi,
        "exclusions": exclusions,
        "May coverage": may_coverage,
        "October coverage": october_coverage,
    }
    for name, mask in supplied_masks.items():
        if mask is not None and (mask.shape != may.shape or mask.dtype != bool):
            raise ValueError(f"{name} requires a boolean mask on the common rCBV grid")

    valid = brain & np.isfinite(may) & np.isfinite(october)
    if exclusions is not None:
        valid &= ~exclusions
    if may_coverage is not None:
        valid &= may_coverage
    if october_coverage is not None:
        valid &= october_coverage

    if not valid.any():
        raise ValueError("No finite common rCBV coverage inside the eligible brain mask")
    roi = reference_roi & valid
    if not roi.any():
        raise ValueError("The reviewed white-matter ROI does not intersect eligible common rCBV coverage")
    return valid, roi
