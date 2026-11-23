"""Multiresolution rigid registration in physical LPS coordinates."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import logging

import numpy as np
import SimpleITK as sitk

from .image_geometry import array_to_image, same_geometry, validate_geometry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistrationOptions:
    """Pixel domains and optimizer settings are configurable per registration pair."""

    initialization: str = "geometry"
    sampling_fraction: float = 0.30
    sampling_seed: int = 1973
    histogram_bins: int = 40
    shrink_factors: tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas_mm: tuple[float, ...] = (2.0, 1.0, 0.0)
    learning_rate: float = 2.0
    minimum_step_mm: float = 0.001
    gradient_tolerance: float = 1e-7
    iterations_per_level: int = 180
    fixed_clip: tuple[float, float] | None = None
    moving_clip: tuple[float, float] | None = None


@dataclass
class RegistrationResult:
    """The transform maps fixed LPS coordinates to moving LPS coordinates."""

    transform: sitk.Euler3DTransform
    metric: str
    metric_value: float
    stop_condition: str
    iterations: int
    options: RegistrationOptions

    def summary(self) -> dict:
        transform = self.transform
        return {
            "direction": "fixed LPS to moving LPS",
            "matrix_row_major": list(transform.GetMatrix()),
            "center_mm_lps": list(transform.GetCenter()),
            "translation_mm_lps": list(transform.GetTranslation()),
            "rotation_degrees_xyz": np.rad2deg(
                [transform.GetAngleX(), transform.GetAngleY(), transform.GetAngleZ()]
            ).tolist(),
            "metric": self.metric,
            "metric_value": self.metric_value,
            "stop_condition": self.stop_condition,
            "total_iterations": self.iterations,
            "options": asdict(self.options),
        }


def _preprocess(
    image: sitk.Image, clip: Sequence[float] | None, mask: sitk.Image | None
) -> tuple[sitk.Image, sitk.Image | None]:
    validate_geometry(image)
    image = sitk.Cast(image, sitk.sitkFloat32)
    values = sitk.GetArrayViewFromImage(image)
    valid = np.isfinite(values)
    if not valid.all():
        if not valid.any():
            raise ValueError("Registration image has no finite data")
        # Keep the filter pyramid finite while rejecting padded data in the metric.
        image = array_to_image(np.where(valid, values, values[valid].min()), image)
        if mask is not None:
            mask = _mask_on_grid(mask, image)
            valid &= sitk.GetArrayViewFromImage(mask) != 0
        mask = array_to_image(valid.astype(np.uint8), image)
    if mask is not None:
        mask = _mask_on_grid(mask, image)
    if clip is not None:
        if len(clip) != 2 or not np.isfinite(clip).all() or clip[0] >= clip[1]:
            raise ValueError("Intensity clipping requires finite lower and upper bounds")
        image = sitk.Clamp(image, lowerBound=float(clip[0]), upperBound=float(clip[1]))
    stats = sitk.MinimumMaximumImageFilter()
    stats.Execute(image)
    if stats.GetMinimum() == stats.GetMaximum():
        raise ValueError("Registration image must have intensity variation after preprocessing")
    return image, mask


def _mask_on_grid(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    if not same_geometry(mask, image):
        raise ValueError("Registration mask must be defined on its corresponding image grid")
    if not np.any(sitk.GetArrayViewFromImage(mask)):
        raise ValueError("Registration mask is empty")
    binary = mask > 0
    binary.CopyInformation(image)
    return binary


def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    metric: str = "mutual_information",
    fixed_mask: sitk.Image | None = None,
    moving_mask: sitk.Image | None = None,
    options: RegistrationOptions | Mapping | None = None,
) -> RegistrationResult:
    """Estimate a six-DOF Euler transform for sampling moving on fixed."""
    if not isinstance(options, RegistrationOptions):
        options = RegistrationOptions(**(options or {}))
    if len(options.shrink_factors) != len(options.smoothing_sigmas_mm):
        raise ValueError("Registration requires one smoothing sigma for each level")
    if not options.shrink_factors or options.shrink_factors[-1] != 1:
        raise ValueError("Registration pyramid must end at full resolution (shrink factor 1)")
    if any(shrink < 1 for shrink in options.shrink_factors):
        raise ValueError("Registration shrink factors must be positive")
    if any(sigma < 0 or not np.isfinite(sigma) for sigma in options.smoothing_sigmas_mm):
        raise ValueError("Registration sigmas must be finite and nonnegative")
    if not 0 < options.sampling_fraction <= 1:
        raise ValueError("Registration sampling_fraction must be in (0, 1]")
    if not (
        options.histogram_bins >= 5
        and options.iterations_per_level >= 1
        and 0 < options.minimum_step_mm < options.learning_rate < np.inf
        and 0 < options.gradient_tolerance < np.inf
    ):
        raise ValueError("Registration histogram, optimizer step sizes or iteration count is invalid")
    fixed, fixed_mask = _preprocess(fixed, options.fixed_clip, fixed_mask)
    moving, moving_mask = _preprocess(moving, options.moving_clip, moving_mask)
    if min(fixed.GetSize() + moving.GetSize()) < 4:
        raise ValueError("Rigid registration requires at least four voxels per axis")
    if options.initialization == "identity":
        transform = sitk.Euler3DTransform()
        center_index = [(length - 1) / 2 for length in fixed.GetSize()]
        transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(center_index))
    elif options.initialization in ("geometry", "moments"):
        method = (
            sitk.CenteredTransformInitializerFilter.GEOMETRY
            if options.initialization == "geometry"
            else sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        transform = sitk.Euler3DTransform(
            sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), method)
        )
    else:
        raise ValueError("Registration initialization must be geometry, moments or identity")

    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(1)
    registration.SetNumberOfWorkUnits(1)
    if metric == "mutual_information":
        registration.SetMetricAsMattesMutualInformation(options.histogram_bins)
    elif metric == "correlation":
        registration.SetMetricAsCorrelation()
    elif metric == "mean_squares":
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Unknown registration metric {metric!r}")
    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration.SetMetricMovingMask(moving_mask)
    if options.sampling_fraction < 1.0:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(options.sampling_fraction, options.sampling_seed)
    else:
        registration.SetMetricSamplingStrategy(registration.NONE)

    usable_levels = [
        (shrink, sigma)
        for shrink, sigma in zip(options.shrink_factors, options.smoothing_sigmas_mm)
        if min(fixed.GetSize() + moving.GetSize()) // shrink >= 4
    ]
    registration.SetShrinkFactorsPerLevel([shrink for shrink, _ in usable_levels])
    registration.SetSmoothingSigmasPerLevel([sigma for _, sigma in usable_levels])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        options.learning_rate, options.minimum_step_mm, options.iterations_per_level,
        gradientMagnitudeTolerance=options.gradient_tolerance,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(transform, inPlace=True)

    iterations = 0

    def count_iteration() -> None:
        nonlocal iterations
        iterations += 1

    registration.AddCommand(sitk.sitkIterationEvent, count_iteration)
    try:
        registration.Execute(fixed, moving)
    except RuntimeError as error:
        raise RuntimeError(
            f"Rigid registration failed ({metric}, {options.initialization} initialization). "
            "Review image coverage, orientation, masks and initialization. "
            f"SimpleITK reported: {error}"
        ) from error
    metric_value = float(registration.GetMetricValue())
    if (
        not np.isfinite(metric_value)
        or abs(metric_value) == np.finfo(np.float64).max
        or registration.GetMetricNumberOfValidPoints() == 0
    ):
        raise RuntimeError("Rigid registration did not produce a finite metric on overlapping data")
    stop_condition = registration.GetOptimizerStopConditionDescription()
    log.info("Rigid registration (%s): %s", metric, stop_condition)
    return RegistrationResult(transform, metric, metric_value, stop_condition, iterations, options)
