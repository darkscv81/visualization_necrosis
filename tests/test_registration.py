"""Known rigid motions and independent alignment checks."""

import numpy as np
import pytest
import SimpleITK as sitk

from necrosis_analysis.image_geometry import array_to_image
from necrosis_analysis.registration import register_rigid
from necrosis_analysis.registration_qa import evaluate_registration


def _lps_lattice(image: sitk.Image) -> np.ndarray:
    indices = np.indices(image.GetSize()[::-1])[::-1].reshape(3, -1).T
    return (indices * image.GetSpacing()) @ np.reshape(image.GetDirection(), (3, 3)).T + image.GetOrigin()


def _build_grid(size: tuple, spacing: tuple, center: np.ndarray) -> sitk.Image:
    angle = np.deg2rad(17)
    direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing(spacing)
    image.SetDirection(direction.ravel())
    image.SetOrigin(center - direction @ (0.5 * (np.asarray(size) - 1) * spacing))
    return image


def _make_phantom(
    grid: sitk.Image,
    center: np.ndarray,
    fixed_to_moving: sitk.Euler3DTransform | None = None,
    multimodal: bool = False,
) -> tuple[sitk.Image, sitk.Image]:
    points = _lps_lattice(grid)
    if fixed_to_moving is not None:
        matrix = np.reshape(fixed_to_moving.GetMatrix(), (3, 3))
        points = (points - center - fixed_to_moving.GetTranslation()) @ matrix + center
    points -= center
    brain = np.sum((points / (21, 17, 15)) ** 2, axis=1) <= 1
    signal = np.zeros(len(points))
    blobs = (
        ((-8, -4, 0), (5, 7, 4), 1.0),
        ((9, -10, 8), (5, 4, 5), 1.5),
        ((-3, 9, -8), (6, 4, 5), 2.4),
        ((9, 6, 4), (4, 6, 2), 2.0),
        ((-11, -7, -8), (4, 2, 2), 1.6),
    )
    for location, widths, amplitude in blobs:
        signal += amplitude * np.exp(-0.5 * np.sum(((points - location) / widths) ** 2, axis=1))
    if multimodal:
        signal = np.log1p(signal) * 40
    return (
        array_to_image(signal.astype(np.float32).reshape(grid.GetSize()[::-1]), grid),
        array_to_image(brain.astype(np.uint8).reshape(grid.GetSize()[::-1]), grid),
    )


@pytest.mark.parametrize("angles_deg,metric,point_tolerance_mm,angle_tolerance_deg", [
    ((0, 0, 0), "correlation", 0.45, 0.6),
    ((4, -3, 5), "correlation", 0.45, 0.6),
    ((4, -3, 5), "mutual_information", 1.0, 1.5),
])
def test_rigid_registration_recovers_synthetic_motion(
    angles_deg: tuple, metric: str, point_tolerance_mm: float, angle_tolerance_deg: float
) -> None:
    center = np.array((41., -23., 18.))
    fixed_grid = _build_grid((41, 39, 31), (1.35, 1.3, 1.7), center)
    moving_grid = _build_grid((44, 43, 28), (1.3, 1.25, 1.85), center + (0.2, -0.4, 0.5))
    truth = sitk.Euler3DTransform()
    truth.SetCenter(center)
    truth.SetTranslation((2.2, -3.3, 1.7))
    truth.SetRotation(*np.deg2rad(angles_deg))
    fixed, fixed_mask = _make_phantom(fixed_grid, center)
    moving, moving_mask = _make_phantom(moving_grid, center, truth, metric == "mutual_information")
    estimate = register_rigid(fixed, moving, metric=metric, options={"sampling_fraction": 1.0})
    assert estimate.iterations > 0
    landmarks = [
        tuple(center + offset) for offset in ((0, 0, 0), (-13, 7, 9), (8, 11, -10), (3, -5, -1))
    ]
    qa = evaluate_registration(
        fixed, moving, estimate.transform, fixed_mask=fixed_mask, moving_mask=moving_mask,
        landmarks={"fixed_lps": landmarks, "moving_lps": [truth.TransformPoint(p) for p in landmarks]},
        thresholds={"min_dice": 0.94, "max_tre_mm": point_tolerance_mm},
    )
    np.testing.assert_allclose(
        estimate.summary()["rotation_degrees_xyz"], angles_deg, atol=angle_tolerance_deg,
    )
    assert qa["before"]["mean_tre_mm"] > 3.0
    assert qa["after"]["max_tre_mm"] < point_tolerance_mm
    assert qa["after"]["dice"] > 0.94
    assert qa["after"]["hausdorff_mm"] <= max(fixed.GetSpacing()) * 1.6
    assert qa["warnings"] == []


def test_registration_flags_misalignment_and_unavailable_metrics() -> None:
    grid = _build_grid((35, 35, 31), (1., 1., 1.5), np.array((9., 8., 7.)))
    fixed, _ = _make_phantom(grid, np.array((9., 8., 7.)))
    qa = evaluate_registration(
        fixed, fixed, sitk.TranslationTransform(3, (80., 0., 0.)),
        thresholds={"min_fixed_coverage": 0.8, "min_dice": 0.85, "max_translation_mm": 8},
    )
    assert qa["before"]["pearson_r"] == pytest.approx(1)
    assert qa["after"]["fixed_coverage_fraction"] == 0
    assert qa["after"]["pearson_r"] is None
    assert any("no finite overlap" in warning for warning in qa["warnings"])
    assert any("dice unavailable" in warning for warning in qa["warnings"])
    assert any("translation_mm" in warning for warning in qa["warnings"])


def test_qa_mask_accepts_verified_rounding_difference_between_dicom_and_nifti() -> None:
    center = np.array((13., -45., 8.))
    grid = _build_grid((35, 35, 31), (1., 1., 1.5), center)
    fixed, mask = _make_phantom(grid, center)
    rounded_mask = sitk.Image(mask)
    rounded_mask.SetOrigin(np.asarray(mask.GetOrigin()) + (0., 4e-6, 0.))
    qa = evaluate_registration(
        fixed, fixed, sitk.Transform(3, sitk.sitkIdentity),
        fixed_mask=rounded_mask, moving_mask=mask,
    )
    assert qa["after"]["dice"] == pytest.approx(1)
    assert qa["after"]["hausdorff_mm"] == pytest.approx(0)
