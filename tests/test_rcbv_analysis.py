import numpy as np
import pytest

from necrosis_analysis.rcbv_analysis import candidate_regions, candidates, change_maps, threshold_sensitivity


def test_change_maps_preserve_sign_and_mask_small_baselines():
    may = np.array([[[2.0, 4.0, 0.0, -1.0, 1.0, 1.0, 1.0]]])
    october = np.array([[[3.0, 3.0, 1.0, 0.0, 0.0, 99.0, -1.0]]])
    valid = np.array([[[True, True, True, True, True, False, True]]])
    maps = change_maps(may, october, valid, baseline_floor=0.01)
    np.testing.assert_allclose(maps["signed"][valid], [1.0, -1.0, 1.0, 1.0, -1.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(maps["magnitude"][valid], [1, 1, 1, 1, 1, 2], atol=1e-12)
    np.testing.assert_allclose(maps["positive"][valid], [1, 0, 1, 1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(maps["negative"][valid], [0, -1, 0, 0, -1, -2], atol=1e-12)
    np.testing.assert_allclose(maps["percent"][valid], [50, -25, np.nan, np.nan, -100, -200], atol=1e-12)
    np.testing.assert_allclose(
        maps["log_ratio"][valid], [np.log(1.5), np.log(0.75), np.nan, np.nan, np.nan, np.nan], atol=1e-12
    )
    assert all(np.isnan(array[0, 0, 5]) for array in maps.values())


def test_regions_are_signed_face_connected_and_use_physical_volume():
    valid = np.ones((2, 3, 5), dtype=bool)
    may = np.ones(valid.shape)
    oct_ = may.copy()
    oct_[0, 0, 0:2] = 1.75
    oct_[0, 1, 2] = 1.75  # Adjacent only diagonally to the two-voxel region.
    oct_[0, 2, 3:5] = 0.25
    oct_[1, 2, 4] = 1.2
    maps = change_maps(may, oct_, valid)
    labels = candidates(maps, valid, valid, absolute=0.3, percent=50.0, min_region_ml=0.024, voxel_ml=0.012)
    assert np.count_nonzero(labels == 1) == np.count_nonzero(labels == -1) == 2
    dose = np.full(valid.shape, 20.0)
    region_dose_coverage = np.ones(valid.shape, dtype=bool)
    region_dose_coverage[0, 0, 1] = False
    rows = candidate_regions(labels, maps, dose=dose, dose_coverage=region_dose_coverage, voxel_ml=0.012)
    assert len(rows) == 2
    assert rows[1]["centroid_z_y_x"] == [0.0, 0.0, 0.5]
    assert rows[1]["volume_ml"] == pytest.approx(0.024, abs=1e-12)
    assert rows[1]["median_signed_change"] == 0.75
    assert rows[1]["dose_evaluable_voxels"] == 1
    sweep = threshold_sensitivity(
        maps, valid, valid, absolute_thresholds=[0.3, 0.8], voxel_ml=0.012, percent=50.0,
        min_region_ml=0.024, dose=dose, dose_coverage=region_dose_coverage, dose_threshold=20,
    )
    assert sweep[0]["candidate_ml"] == pytest.approx(0.048, abs=1e-12)
    assert sweep[0]["high_dose_overlap_ml"] == pytest.approx(0.036, abs=1e-12)
    assert sweep[1]["candidate_ml"] == 0
    assert sweep[1]["median_dose"] is None


def test_robust_threshold_uses_reference_mad():
    may = np.ones((1, 1, 7))
    october = np.array([[[0.9, 0.95, 1.0, 1.05, 1.1, 0.5, 1.12]]])
    valid = np.ones(may.shape, dtype=bool)
    roi = np.arange(7).reshape(may.shape) < 5
    labels = candidates(change_maps(may, october, valid), valid, roi, robust_sigma=2)
    np.testing.assert_array_equal(labels.ravel(), [0, 0, 0, 0, 0, -1, 0])
    with pytest.raises(ValueError, match="zero MAD"):
        candidates(change_maps(may, may, valid), valid, roi, robust_sigma=2)
