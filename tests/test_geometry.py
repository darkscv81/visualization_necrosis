"""Independent coordinate and interpolation checks on sampled image grids."""

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from necrosis_analysis.image_geometry import (
    array_to_image,
    compose_transforms,
    geometry_summary,
    index_to_lps,
    lps_to_index,
    lps_to_ras,
    resample_mask,
    resample_scalar,
    same_geometry,
    validate_geometry,
    voxel_to_ras_affine,
)


def oblique_grid() -> sitk.Image:
    image = sitk.Image((9, 7, 5), sitk.sitkFloat64)
    image.SetOrigin((32, -55, 21))
    image.SetSpacing((1.2, 2, 3))
    image.SetDirection((0, -0.8, 0.6, 1, 0, 0, 0, 0.6, 0.8))
    return image


def test_voxel_coordinates_agree_with_independent_nifti_affine() -> None:
    image = oblique_grid()
    index = (2.5, 1, 0.5)
    # Direction columns x=(0, 1, 0), y=(-0.8, 0, 0.6), z=(0.6, 0, 0.8).
    expected_lps = (31.3, -52, 23.4)
    np.testing.assert_allclose(index_to_lps(image, index), expected_lps, atol=1e-10)
    np.testing.assert_allclose(lps_to_index(image, expected_lps), index, atol=1e-10)
    np.testing.assert_allclose(lps_to_ras(expected_lps), (-31.3, 52, 23.4), atol=1e-10)
    nifti_point = nib.affines.apply_affine(voxel_to_ras_affine(image), index)
    np.testing.assert_allclose(nifti_point, (-31.3, 52, 23.4), atol=1e-10)
    assert geometry_summary(image)["voxel_volume_mm3"] == pytest.approx(7.2)


def test_geometry_rejects_shear() -> None:
    image = oblique_grid()
    image.SetDirection((1, 0.2, 0, 0, 1, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="orthonormal"):
        validate_geometry(image)


def test_composition_uses_physical_application_order() -> None:
    translation = sitk.TranslationTransform(3, (3, 1, -2))
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0, 0, np.pi / 2)
    composed = compose_transforms(compose_transforms(translation), rotation)
    np.testing.assert_allclose(composed.TransformPoint((0, 0, 5)), (-1, 3, 3), atol=1e-10)
    np.testing.assert_allclose(composed.GetInverse().TransformPoint((-1, 3, 3)), (0, 0, 5), atol=1e-10)


def test_scalar_linear_interpolation_respects_missing_support_and_physical_shift() -> None:
    reference = oblique_grid()
    values = np.tile(np.arange(9, dtype=float), (5, 7, 1))
    values[:, :, 4] = np.nan
    moving = array_to_image(values, reference)
    # +0.6 mm in LPS y moves half a voxel along the image's x axis.
    transform = sitk.TranslationTransform(3, (0, 0.6, 0))
    sampled, validity = resample_scalar(moving, reference, transform)
    assert same_geometry(sampled, reference) and same_geometry(validity, reference)
    line = sitk.GetArrayFromImage(sampled)[2, 3]
    np.testing.assert_allclose(
        line, [0.5, 1.5, 2.5, np.nan, np.nan, 5.5, 6.5, 7.5, np.nan], atol=1e-10,
    )
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(validity)[2, 3], [1, 1, 1, 0, 0, 1, 1, 1, 0],
    )


def test_mask_resampling_preserves_labels_and_masks_outside_grid() -> None:
    reference = oblique_grid()
    labels = np.zeros((5, 7, 9), np.uint16)
    labels[2, 3, 3] = 7
    mask = array_to_image(labels, reference)
    sampled = resample_mask(mask, reference, sitk.TranslationTransform(3, (0, 1.2, 0)))
    assert sampled.GetPixelID() == sitk.sitkUInt16
    assert sampled[2, 3, 2] == 7
    assert sampled[3, 3, 2] == 0
    assert sampled[8, 3, 2] == 0
