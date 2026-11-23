"""Cross-check the pipeline fixture through the production readers and registration."""

from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

from necrosis_analysis.brain_segmentation import analysis_masks
from necrosis_analysis.dicom_io import read_rtdose, read_volume
from necrosis_analysis.image_geometry import resample_scalar, same_geometry
from necrosis_analysis.normalization import normalize
from necrosis_analysis.rcbv_analysis import change_maps
from necrosis_analysis.registration import register_rigid
from necrosis_analysis.registration_qa import evaluate_registration
from necrosis_analysis.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="module")
def fixture_config(tmp_path_factory):
    return generate_synthetic_dataset(tmp_path_factory.mktemp("phantom"))


def test_fixture_has_known_dose_geometry_and_scale(fixture_config):
    paths = fixture_config["paths"]
    dose = read_rtdose(paths["dose"])
    ct = read_volume(paths["ct"], modality="CT")
    header = pydicom.dcmread(next(Path(paths["dose"]).glob("*.dcm")))
    # Reconstruct the test point directly from the DICOM record.
    x, y, z = 17, 16, 13
    row, column = np.reshape(np.asarray(header.ImageOrientationPatient, dtype=float), (2, 3))
    spacing_y, spacing_x = np.asarray(header.PixelSpacing, dtype=float)
    position = (
        np.asarray(header.ImagePositionPatient, dtype=float)
        + row * (x * spacing_x) + column * (y * spacing_y)
        + np.cross(row, column) * float(header.GridFrameOffsetVector[z])
    )
    assert dose.image.TransformIndexToPhysicalPoint((x, y, z)) == pytest.approx(position, abs=1e-6)
    assert dose.image.GetSize() == (36, 34, 28)
    assert dose.frame_uid == ct.frame_uid
    assert not same_geometry(ct.image, dose.image)
    assert float(header.DoseGridScaling) == pytest.approx(0.004)
    assert int(header.pixel_array[z, y, x]) == 10763
    assert dose.image.GetPixel((x, y, z)) == pytest.approx(43.052, abs=1e-5)


def test_fixture_rigid_transforms_and_independent_masks(fixture_config):
    paths = fixture_config["paths"]
    fixed = read_volume(paths["may_anatomy"], modality="MR").image
    fixed_mask = sitk.ReadImage(paths["brain_mask"])
    assert fixed.GetSize() == (52, 48, 42)
    assert same_geometry(fixed, fixed_mask)
    assert abs(np.asarray(fixed.GetDirection()).reshape(3, 3)[1, 0]) > 0.1

    may_rcbv = read_volume(paths["may_rcbv"], modality="MR").image
    october_rcbv = read_volume(paths["october_rcbv"], modality="MR").image
    assert may_rcbv.GetSize()[2] == 20
    assert october_rcbv.GetSize()[2] == 18

    for name, anatomy, modality in [("ct", "ct", "CT"), ("october", "october_anatomy", "MR")]:
        moving = read_volume(paths[anatomy], modality=modality).image
        moving_mask = sitk.ReadImage(paths[f"{name}_brain_mask"])
        settings = fixture_config["registration"]
        result = register_rigid(
            fixed, moving, metric=settings["metrics"][name],
            options=settings["options"][name], fixed_mask=fixed_mask, moving_mask=moving_mask,
        )
        truth = fixture_config["synthetic_truth"]
        qa = evaluate_registration(
            fixed, moving, result.transform, fixed_mask=fixed_mask, moving_mask=moving_mask,
            landmarks=fixture_config["landmarks"][name],
        )
        assert qa["after"]["max_tre_mm"] < truth["tre_tolerance_mm"]
        assert qa["after"]["mean_tre_mm"] < qa["before"]["mean_tre_mm"] / 4
        assert qa["after"]["dice"] > 0.96

        expected = truth["fixed_to_moving"][name]
        estimate = np.asarray(result.transform.GetMatrix()).reshape(3, 3)
        rotation_error = Rotation.from_matrix(estimate @ np.asarray(expected["matrix"]).T)
        assert np.rad2deg(rotation_error.magnitude()) < 1.5
        center = expected["center_lps_mm"]
        shift = np.asarray(result.transform.TransformPoint(center)) - center
        assert shift == pytest.approx(expected["translation_mm"], abs=0.5)


def test_known_sampling_maps_preserve_perfusion_changes_and_dose(fixture_config):
    paths = fixture_config["paths"]
    truth = fixture_config["synthetic_truth"]
    fixed = read_volume(paths["may_anatomy"]).image
    native = {session: read_volume(paths[f"{session}_rcbv"]).image for session in ("may", "october")}
    dose = read_rtdose(paths["dose"]).image

    transforms = {}
    for name, pose in truth["fixed_to_moving"].items():
        matrix = np.asarray(pose["matrix"]).ravel().tolist()
        transforms[name] = sitk.AffineTransform(matrix, pose["translation_mm"], pose["center_lps_mm"])
    may = sitk.GetArrayFromImage(resample_scalar(native["may"], fixed)[0])
    october = sitk.GetArrayFromImage(resample_scalar(native["october"], fixed, transforms["october"])[0])
    dose_on_may = resample_scalar(dose, fixed, transforms["ct"])[0]
    brain = sitk.GetArrayFromImage(sitk.ReadImage(paths["brain_mask"])) != 0
    reference_roi = sitk.GetArrayFromImage(sitk.ReadImage(paths["reference_roi"])) != 0
    valid, roi = analysis_masks(brain, reference_roi, may, october)
    may_norm, october_norm, stats = normalize(may, october, valid, roi)
    changes = change_maps(may_norm, october_norm, valid)["signed"]

    assert 0 < valid.sum() < brain.sum()
    assert stats["factors"]["may"] == pytest.approx(truth["may_rcbv_nominal_reference"], rel=0.02)
    assert stats["factors"]["october"] == pytest.approx(truth["october_rcbv_nominal_reference"], rel=0.02)
    decrease = fixed.TransformPhysicalPointToIndex(truth["decrease_center_lps_mm"])
    increase = fixed.TransformPhysicalPointToIndex(truth["increase_center_lps_mm"])
    assert changes[decrease[::-1]] == pytest.approx(-0.52, abs=0.04)
    assert changes[increase[::-1]] == pytest.approx(0.46, abs=0.04)
    assert dose_on_may.GetPixel(decrease) == pytest.approx(80.55, abs=0.30)
    assert dose_on_may.GetPixel(increase) == pytest.approx(12.32, abs=0.30)


def test_repeat_generation_preserves_fixture_files(tmp_path):
    first = generate_synthetic_dataset(tmp_path)
    file = next(Path(first["paths"]["dose"]).glob("*.dcm"))
    modified = file.stat().st_mtime_ns
    assert generate_synthetic_dataset(tmp_path) == first
    assert file.stat().st_mtime_ns == modified
