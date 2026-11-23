"""Independently encoded DICOM examples with analytically known geometry and values."""

import hashlib
import json

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, MRImageStorage, RTDoseStorage, generate_uid

from necrosis_analysis.dicom_io import (
    find_anatomical_candidates,
    read_rtdose,
    read_volume,
    scan_series,
)


def _object(path, pixels, modality, sop_class, study, series, frame):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(path, {}, file_meta=meta, preamble=bytes(128))
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series
    ds.StudyInstanceUID = study
    ds.FrameOfReferenceUID = frame
    ds.StudyDate = "20000517"
    ds.PatientName = "Synthetic^Test"
    ds.PatientID = "placeholder"
    ds.Modality = modality
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.BitsAllocated = ds.BitsStored = pixels.dtype.itemsize * 8
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = int(pixels.dtype.kind == "i")
    ds.PixelData = pixels.tobytes()
    return ds


def _mr_slices(path, positions):
    study, series, frame = [generate_uid() for _ in range(3)]
    for index, x in enumerate(positions):
        file = path / f"plane-{index}.dcm"
        pixels = np.full((2, 3), index + 1, dtype="<i2")
        pixels[0, 0] = -3000
        ds = _object(file, pixels, "MR", MRImageStorage, study, series, frame)
        ds.SeriesDescription = "Axial T1 Post"
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        ds.ImagePositionPatient = [x, -17.5, 30]
        ds.PixelSpacing = [2.5, 1.25]
        ds.RescaleSlope = 2
        ds.RescaleIntercept = -1
        ds.PixelPaddingValue = -3000
        ds.InstanceNumber = index
        ds.save_as(file, enforce_file_format=True)
    return series


def test_classic_mr_physical_sort_padding_scaling_and_source_preservation(tmp_path):
    series = _mr_slices(tmp_path, [20.0, 12.0, 16.0])
    (tmp_path / "notes.txt").write_text("Non-DICOM companion notes.")
    inputs = sorted(tmp_path.iterdir())
    before = [hashlib.sha256(path.read_bytes()).digest() for path in inputs]

    listing = scan_series(tmp_path)
    assert [(entry.series_uid, entry.frame_count) for entry in listing] == [(series, 3)]
    assert find_anatomical_candidates(tmp_path, study_uid=listing[0].study_uid) == listing
    loaded = read_volume(tmp_path, series_uid=series, modality="MR")
    assert loaded.image.GetSize() == (3, 2, 3)
    assert loaded.image.GetSpacing() == (1.25, 2.5, 4.0)
    assert loaded.image.GetOrigin() == (12.0, -17.5, 30.0)
    assert loaded.image.TransformIndexToPhysicalPoint((2, 1, 2)) == (20.0, -15.0, 32.5)
    voxels = sitk.GetArrayFromImage(loaded.image)
    np.testing.assert_array_equal(voxels[:, 1, 2], [3, 5, 1])
    assert np.isnan(voxels[:, 0, 0]).all()
    assert loaded.metadata["value_scaling"][0]["method"] == "modality_rescale"
    assert "Synthetic^Test" not in json.dumps(loaded.to_dict(), allow_nan=False)
    assert [hashlib.sha256(path.read_bytes()).digest() for path in inputs] == before


def test_native_dicom_without_part_10_header(tmp_path):
    _mr_slices(tmp_path, [0.0, 4.0])
    path = tmp_path / "plane-0.dcm"
    ds = pydicom.dcmread(path)
    ds.preamble = None
    ds.file_meta = FileMetaDataset()
    pydicom.dcmwrite(path, ds, implicit_vr=False, little_endian=True)
    assert read_volume(tmp_path).image.GetPixel(2, 1, 0) == 1.0


@pytest.mark.parametrize(
    ("positions", "message"),
    [([0.0, 4.0, 4.0], "duplicated"), ([0.0, 4.0, 12.0], "nonuniform")],
)
def test_reject_duplicate_or_missing_slice_centers(tmp_path, positions, message):
    _mr_slices(tmp_path, positions)
    with pytest.raises(ValueError, match=message):
        read_volume(tmp_path)


def test_enhanced_mr_reads_per_frame_geometry_and_real_world_values(tmp_path):
    path = tmp_path / "enhanced.dcm"
    pixels = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="<u2")
    study, series, frame = [generate_uid() for _ in range(3)]
    ds = _object(path, pixels, "MR", EnhancedMRImageStorage, study, series, frame)
    ds.NumberOfFrames = 2
    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = [orientation]
    measures = Dataset()
    measures.PixelSpacing = [3.0, 2.0]
    measures.SliceThickness = 5.0
    shared.PixelMeasuresSequence = [measures]
    mapping = Dataset()
    mapping.RealWorldValueFirstValueMapped = 0
    mapping.RealWorldValueLastValueMapped = 7
    mapping.RealWorldValueSlope = 0.5
    mapping.RealWorldValueIntercept = 10.0
    units = Dataset()
    units.CodeValue = "1"
    units.CodingSchemeDesignator = "UCUM"
    units.CodeMeaning = "unity"
    mapping.MeasurementUnitsCodeSequence = [units]
    shared.RealWorldValueMappingSequence = [mapping]
    ds.SharedFunctionalGroupsSequence = [shared]
    ds.PerFrameFunctionalGroupsSequence = []
    for z in (-10, -15):
        group = Dataset()
        position = Dataset()
        position.ImagePositionPatient = [4, -5, z]
        group.PlanePositionSequence = [position]
        ds.PerFrameFunctionalGroupsSequence.append(group)
    ds.save_as(path, enforce_file_format=True)

    loaded = read_volume(tmp_path)
    assert loaded.image.GetOrigin() == (4.0, -5.0, -15.0)
    assert loaded.image.GetSpacing() == (2.0, 3.0, 5.0)
    voxels = sitk.GetArrayFromImage(loaded.image)
    assert voxels[0, 0, 0] == 12.5
    assert np.isnan(voxels[0, 1, 1])
    assert voxels[1, 1, 1] == 12.0
    assert loaded.metadata["value_scaling"][0]["units"]["code"] == "1"


def _dose(path, *, orientation, origin, offsets):
    pixels = (np.arange(24, dtype="<u4") * 200).reshape(3, 2, 4)
    study, series, frame = [generate_uid() for _ in range(3)]
    ds = _object(path, pixels, "RTDOSE", RTDoseStorage, study, series, frame)
    ds.NumberOfFrames = 3
    ds.ImageOrientationPatient = orientation
    ds.ImagePositionPatient = origin
    ds.PixelSpacing = [2.75, 1.5]
    ds.FrameIncrementPointer = Tag(0x3004000C)
    ds.GridFrameOffsetVector = offsets
    ds.DoseGridScaling = 0.0005
    ds.DoseUnits = "GY"
    ds.DoseType = "EFFECTIVE"
    ds.DoseSummationType = "MULTI_PLAN"
    ds.DoseComment = "Independent synthetic dose test"
    ref = Dataset()
    ref.ReferencedSOPInstanceUID = generate_uid()
    ds.ReferencedRTPlanSequence = [ref]
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_independent_oblique_descending_rtdose_scaling_and_lps_coordinates(tmp_path):
    ds = _dose(
        tmp_path / "dose",
        orientation=[0.6, 0.8, 0, 0, 0, 1],
        origin=[30.0, -14.0, 2.0],
        offsets=[0.0, -4.0, -8.0],
    )
    loaded = read_rtdose(tmp_path)
    assert loaded.frame_uid == ds.FrameOfReferenceUID
    assert loaded.metadata["dose_units"] == "GY"
    assert loaded.metadata["dose_type"] == "EFFECTIVE"
    assert loaded.metadata["referenced_rt_plans"][0]["sop_instance_uid"] == (
        ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
    )
    assert loaded.metadata["grid_frame_offset_encoding"] == "relative_to_first_frame"
    assert loaded.image.GetSpacing() == (1.5, 2.75, 4.0)
    assert loaded.image.GetOrigin() == pytest.approx((23.6, -9.2, 2.0), abs=1e-8)
    assert loaded.image.TransformIndexToPhysicalPoint((3, 1, 2)) == pytest.approx(
        (32.7, -10.4, 4.75), abs=1e-8
    )
    # DICOM frame 2, row 1, column 3 encodes sample 23, scaled from 4600 to 2.3 Gy.
    assert loaded.image.GetPixel(3, 1, 0) == pytest.approx(2.3, abs=1e-6)
    assert loaded.image.GetPixel(3, 1, 2) == pytest.approx(0.7, abs=1e-6)
    assert not loaded.warnings


def test_axial_absolute_z_offsets_and_nonuniform_grid_rejection(tmp_path):
    path = tmp_path / "dose.dcm"
    ds = _dose(
        path,
        orientation=[1, 0, 0, 0, 1, 0],
        origin=[-5.0, 7.0, 25.0],
        offsets=[25.0, 29.0, 33.0],
    )
    loaded = read_rtdose(tmp_path)
    assert loaded.metadata["grid_frame_offset_encoding"] == "absolute_axial_patient_z"
    assert loaded.image.GetOrigin() == (-5.0, 7.0, 25.0)
    assert loaded.image.TransformIndexToPhysicalPoint((3, 1, 2)) == (-0.5, 9.75, 33.0)
    assert loaded.image.GetPixel(3, 1, 2) == pytest.approx(2.3, abs=1e-6)

    ds.GridFrameOffsetVector = [25.0, 28.0, 33.0]
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="nonuniform"):
        read_rtdose(tmp_path)


def test_dose_missing_scaling_is_rejected(tmp_path):
    path = tmp_path / "dose.dcm"
    ds = _dose(
        path, orientation=[1, 0, 0, 0, 1, 0], origin=[0, 0, 0], offsets=[0, 1, 2]
    )
    del ds.DoseGridScaling
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="DoseGridScaling"):
        read_rtdose(tmp_path)
