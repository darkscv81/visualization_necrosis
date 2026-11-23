import numpy as np
import pytest

from necrosis_analysis.brain_segmentation import analysis_masks, load_mask
from necrosis_analysis.normalization import normalize


def test_roi_and_common_coverage():
    may = np.full((1, 2, 4), 2.0)
    october = np.full((1, 2, 4), 6.0)
    october[0, 1, 2] = np.nan
    brain = (np.arange(8) < 7).reshape(may.shape)
    roi = (np.arange(8) < 3).reshape(may.shape)
    exclusions = (np.arange(8) == 0).reshape(may.shape)
    coverage = (np.arange(8) != 4).reshape(may.shape)
    valid, roi = analysis_masks(brain, roi, may, october, exclusions=exclusions, may_coverage=coverage)
    np.testing.assert_array_equal(np.flatnonzero(valid), [1, 2, 3, 5])
    np.testing.assert_array_equal(np.flatnonzero(roi), [1, 2])


@pytest.mark.parametrize("method, factors, oct_ref", [
    ("white_matter", (2.0, 6.0), 1.0),
    ("shared_may", (2.0, 2.0), 3.0),
    ("already_normalized", (1.0, 1.0), 6.0),
])
def test_normalization_strategies(method, factors, oct_ref):
    may = np.array([[[1.0, 2.0, 3.0, 5.0, 9.0]]])
    october = np.array([[[3.0, 6.0, 9.0, 18.0, 99.0]]])
    valid = np.array([[[True, True, True, True, False]]])
    roi = np.array([[[True, True, True, False, False]]])
    normalized_may, normalized_october, qa = normalize(may, october, valid, roi, method=method, min_roi_voxels=3)
    assert qa["factors"] == dict(zip(("may", "october"), factors))
    assert qa["normalized_reference"]["october"]["median"] == oct_ref
    assert qa["raw"]["may"]["n"] == 4
    assert qa["raw_reference"]["may"]["iqr"] == 1
    assert np.isnan(normalized_may[0, 0, 4])
    np.testing.assert_allclose(normalized_october[valid], october[valid] / factors[1], atol=1e-12)
    if method == "white_matter":
        assert normalized_october[0, 0, 3] - normalized_may[0, 0, 3] == 0.5


def test_normalization_rejects_small_roi_and_nonpositive_median():
    valid = np.ones((2, 3, 4), dtype=bool)
    may = np.ones(valid.shape)
    with pytest.raises(ValueError, match="fewer"):
        normalize(may, may, valid, valid, min_roi_voxels=25)
    with pytest.raises(ValueError, match="medians must be positive"):
        normalize(may, -may, valid, valid)


def test_mask_resampling_uses_physical_coordinates(tmp_path):
    import SimpleITK as sitk

    values = np.zeros((3, 3, 3), dtype=np.uint8)
    values[1, 1, 1] = 255
    image = sitk.GetImageFromArray(values)
    image.SetOrigin((15.0, 10.0, -18.0))
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(path))
    fixed = sitk.Image([3, 3, 3], sitk.sitkFloat32)
    fixed.CopyInformation(image)
    fixed.SetOrigin((15.0, 12.0, -18.0))
    mask = load_mask(path, fixed)
    np.testing.assert_array_equal(np.argwhere(mask), [[1, 1, 0]])
