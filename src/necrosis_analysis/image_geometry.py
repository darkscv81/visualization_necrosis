"""Physical geometry and resampling for scalar 3D images in DICOM LPS."""

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk


def validate_geometry(image: sitk.Image) -> None:
    """Require a nonempty, scalar 3D image with an orthonormal direction matrix."""
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar 3D image")
    if min(image.GetSize()) < 1 or not all(np.isfinite(image.GetOrigin())):
        raise ValueError("Image must have a nonempty grid and a finite LPS origin")
    spacing = np.asarray(image.GetSpacing())
    if not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Voxel spacing must be finite and positive")
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("Image direction cosines must be orthonormal")


def geometry_summary(image: sitk.Image) -> dict:
    """Describe the grid in image-index order, with bounds at voxel centers."""
    validate_geometry(image)
    size = np.asarray(image.GetSize()) - 1
    corners = np.asarray([
        image.TransformIndexToPhysicalPoint((i, j, k))
        for i in (0, int(size[0]))
        for j in (0, int(size[1]))
        for k in (0, int(size[2]))
    ])
    return {
        "coordinates": "DICOM LPS",
        "size_xyz": list(image.GetSize()),
        "spacing_mm_xyz": list(image.GetSpacing()),
        "origin_mm_lps": list(image.GetOrigin()),
        "direction_lps_row_major": list(image.GetDirection()),
        "voxel_volume_mm3": float(np.prod(image.GetSpacing())),
        "center_bounds_mm_lps": [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
    }


def same_geometry(first: sitk.Image, second: sitk.Image, atol: float = 1e-5) -> bool:
    """Check whether two images occupy the same sampled grid."""
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), rtol=0, atol=atol)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), rtol=0, atol=atol)
        and np.allclose(first.GetDirection(), second.GetDirection(), rtol=0, atol=atol)
    )


def array_to_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Attach reference geometry to an array indexed (z, y, x)."""
    if array.shape != reference.GetSize()[::-1]:
        raise ValueError("Array shape does not match the image grid in (z, y, x) order")
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def index_to_lps(image: sitk.Image, index_xyz: Sequence[float]) -> tuple[float, ...]:
    """Convert a continuous (x, y, z) image index to physical LPS millimeters."""
    return image.TransformContinuousIndexToPhysicalPoint(tuple(map(float, index_xyz)))


def lps_to_index(image: sitk.Image, point_lps: Sequence[float]) -> tuple[float, ...]:
    """Convert physical LPS millimeters to a continuous (x, y, z) image index."""
    return image.TransformPhysicalPointToContinuousIndex(tuple(map(float, point_lps)))


def lps_to_ras(coordinates: Sequence[float]) -> np.ndarray:
    """Reflect the left/right and anterior/posterior coordinates; also its inverse."""
    return np.asarray(coordinates, dtype=float) * (-1.0, -1.0, 1.0)


def voxel_to_ras_affine(image: sitk.Image) -> np.ndarray:
    """Map NIfTI-style (x, y, z) voxel indices to physical RAS millimeters."""
    validate_geometry(image)
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    affine[:2, :] *= -1
    return affine


def compose_transforms(*transforms: sitk.Transform) -> sitk.Transform:
    """Compose physical point maps, taking them in the order they are applied."""
    if any(transform.GetDimension() != 3 for transform in transforms):
        raise ValueError("Only 3D transforms can be composed")
    if not transforms:
        return sitk.Transform(3, sitk.sitkIdentity)
    composite = sitk.CompositeTransform(list(reversed(transforms)))
    composite.FlattenTransform()
    return composite


def _interpolation_support(
    valid: np.ndarray, moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform
) -> np.ndarray:
    if valid.all():
        # A constant image gives identical coverage for either interpolator. A
        # byte image avoids allocating floating-point volumes for extent QA.
        source_validity = array_to_image(valid.astype(np.uint8), moving)
        support = sitk.Resample(
            source_validity, reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
        return sitk.GetArrayFromImage(support).astype(bool)
    source_validity = array_to_image(valid.astype(np.float32), moving)
    support = sitk.Resample(
        source_validity, reference, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    return sitk.GetArrayViewFromImage(support) >= 1.0 - 1e-6


def resample_validity(
    moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform
) -> np.ndarray:
    """Return where trilinear interpolation of finite source data is supported."""
    validate_geometry(moving)
    validate_geometry(reference)
    valid = np.isfinite(sitk.GetArrayViewFromImage(moving))
    return _interpolation_support(valid, moving, reference, transform)


def resample_scalar(
    moving: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform | None = None,
    *,
    allow_negative: bool = True,
) -> tuple[sitk.Image, sitk.Image]:
    """Trilinearly sample moving at transform(reference LPS) and return value and validity images.

    Outside the moving image or where the interpolation footprint includes missing
    data, the returned value is NaN. Set allow_negative=False only when that
    modality defines negative source values as invalid.
    """
    validate_geometry(moving)
    validate_geometry(reference)
    if transform is None:
        transform = compose_transforms()
    values = sitk.GetArrayViewFromImage(moving)
    valid = np.isfinite(values)
    if not allow_negative:
        valid &= values >= 0
    pixel_type = sitk.sitkFloat64 if moving.GetPixelID() == sitk.sitkFloat64 else sitk.sitkFloat32
    clean_values = moving if valid.all() else array_to_image(np.where(valid, values, 0), moving)
    sampled = sitk.Resample(clean_values, reference, transform, sitk.sitkLinear, 0.0, pixel_type)
    complete = _interpolation_support(valid, moving, reference, transform)
    result = sitk.GetArrayFromImage(sampled)
    result[~complete] = np.nan
    return array_to_image(result, reference), array_to_image(complete.astype(np.uint8), reference)


def resample_mask(
    moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform | None = None
) -> sitk.Image:
    """Resample a binary or categorical integer mask by nearest neighbor."""
    validate_geometry(moving)
    validate_geometry(reference)
    if transform is None:
        transform = compose_transforms()
    if not np.issubdtype(sitk.GetArrayViewFromImage(moving).dtype, np.integer):
        raise ValueError("A categorical mask must have an integer pixel type")
    return sitk.Resample(
        moving, reference, transform, sitk.sitkNearestNeighbor, 0, moving.GetPixelID()
    )
