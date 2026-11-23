"""Deterministic, independently sampled DICOM fixture for the full pipeline.

Coordinates in the phantom and the ground truth are in DICOM patient LPS
millimeters. Only the generator uses the analytic ground truth. Image loading,
registration and quantitative resampling use the files written by this module.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from numpy.typing import NDArray
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    RTDoseStorage,
)
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class _Grid:
    size: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: NDArray[np.float64]
    direction: NDArray[np.float64]

    def coordinates(self) -> NDArray[np.float64]:
        z, y, x = np.indices(self.size[::-1], dtype=float)
        local = np.stack((x * self.spacing[0], y * self.spacing[1], z * self.spacing[2]))
        return self.origin[:, None, None, None] + np.einsum("ij,j...->i...", self.direction, local)


@dataclass(frozen=True)
class _Pose:
    center: NDArray[np.float64]
    angles: tuple[float, float, float]
    translation: tuple[float, float, float]

    @property
    def matrix(self) -> NDArray[np.float64]:
        return Rotation.from_euler("xyz", self.angles, degrees=True).as_matrix()

    def canonical(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return May-centered positions from the sampled session's world coordinates."""
        moving_center = self.center + self.translation
        return np.einsum("ij,j...->i...", self.matrix.T, coordinates - moving_center[:, None, None, None])

    def map_points(self, fixed: NDArray[np.float64]) -> NDArray[np.float64]:
        return (fixed - self.center) @ self.matrix.T + self.center + self.translation

    def truth(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "center_lps_mm": self.center.tolist(),
            "translation_mm": list(self.translation),
            "euler_xyz_degrees": list(self.angles),
        }


def _grid(
    size: tuple[int, int, int],
    spacing: tuple[float, float, float],
    center: NDArray[np.float64],
    angles: tuple[float, float, float],
) -> _Grid:
    direction = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    origin = center - direction @ ((np.asarray(size) - 1) * spacing / 2)
    return _Grid(size, spacing, origin, direction)


def _ellipsoid(
    coordinates: NDArray[np.float64],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> NDArray[np.float64]:
    displacement = (coordinates - np.asarray(center)[:, None, None, None]) / np.asarray(radii)[:, None, None, None]
    return np.sum(np.square(displacement), axis=0)


def _gaussian(
    coordinates: NDArray[np.float64],
    center: tuple[float, float, float],
    widths: tuple[float, float, float],
) -> NDArray[np.float64]:
    return np.exp(-0.5 * _ellipsoid(coordinates, center, widths))


_DECREASE_CENTER = (13.0, 4.0, -3.0)
_INCREASE_CENTER = (-10.0, -9.0, 7.0)
_REFERENCE_CENTER = (-14.0, 6.0, -6.0)


def _ventricles(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    left = _gaussian(coordinates, (-5.2, -2.4, 1.8), (2.4, 8.1, 2.7))
    right = _gaussian(coordinates, (4.1, -2.1, 0.7), (2.6, 7.4, 3.1))
    return np.maximum(left, right)


def _brain_mask(coordinates: NDArray[np.float64]) -> NDArray[np.bool_]:
    head_distance = _ellipsoid(coordinates, (0, 0, 0), (25, 23, 24))
    return (head_distance <= 1.0) & (_ventricles(coordinates) < 0.65)


def _anatomy(coordinates: NDArray[np.float64], modality: str) -> NDArray[np.float64]:
    x, y, z = coordinates
    brain_distance = _ellipsoid(coordinates, (0, 0, 0), (25, 23, 24))
    skull_outer = _ellipsoid(coordinates, (0.7, 0.3, -0.5), (28.7, 26.5, 27.6))
    scalp_outer = _ellipsoid(coordinates, (0.7, 0.3, -0.5), (30.7, 28.4, 29))
    ventricles = _ventricles(coordinates)
    caudate = _gaussian(coordinates, (7, 2, -7), (3, 4, 5))
    posterior = _gaussian(coordinates, (-10, 9, -6), (4, 6, 4))
    asymmetry = _gaussian(coordinates, (13, -11, 7), (5, 3, 6))
    striation = np.sin(0.48 * x + 0.22 * z) * np.cos(0.40 * y - 0.26 * z)
    striation += 0.52 * np.sin(0.14 * x + 0.36 * y + 0.31 * z) * np.cos(0.12 * x + 0.31 * z)
    gray_ribbon = np.exp(-np.square((brain_distance - 0.86) / 0.1))
    tissue = (brain_distance <= 1.0)
    bone = (brain_distance > 1.08) & (skull_outer <= 1.0)
    scalp = (skull_outer > 1.0) & (scalp_outer <= 1.0)
    values = 630 - 450 * ventricles + 170 * caudate - 160 * posterior + 130 * asymmetry + 65 * striation
    values -= 115 * gray_ribbon
    if modality == "CT":
        tissue_hu = 2.5 + 0.057 * values + 1.9 * np.tanh(striation)
        return np.select(
            [tissue, bone, scalp, skull_outer <= 1.0],
            [tissue_hu, 900 + 80 * np.cos(y / 9), 30, 11],
            default=-1000,
        )
    return np.select([tissue, scalp, bone], [values, 210, 35], default=0)


def _perfusion(coordinates: NDArray[np.float64], followup: bool) -> NDArray[np.float64]:
    x, y, z = coordinates
    base = 1.0 + 0.045 * np.sin(x * 0.16) * np.cos(z * 0.18) + 0.025 * np.sin(y * 0.20)
    base += 0.22 * _gaussian(coordinates, (12, -5, 9), (4, 7, 5))
    if followup:
        decrease = _gaussian(coordinates, _DECREASE_CENTER, (4.5, 4.2, 4.7))
        increase = _gaussian(coordinates, _INCREASE_CENTER, (4.5, 4.2, 4.7))
        relative_change = 1.0 - 0.55 * decrease + 0.52 * increase
        return np.where(_brain_mask(coordinates), 3.6 * base * relative_change, 0.0)
    return np.where(_brain_mask(coordinates), 2.4 * base, 0.0)


def _dose(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    hot_spot = _gaussian(coordinates, _DECREASE_CENTER, (9.5, 10.5, 9.2))
    lower_dose = _gaussian(coordinates, (2, 3, 1), (18, 14, 14))
    return 5.0 + 66.0 * hot_spot + 13.0 * lower_dose


def _uid(label: str) -> str:
    return f"2.25.{uuid.uuid5(uuid.NAMESPACE_URL, 'urn:longitudinal-rcbv-synthetic:' + label).int}"


def _decimal(value: float) -> str:
    return f"{value:.10g}"


def _decimals(values: NDArray[np.float64] | tuple[float, ...]) -> list[str]:
    return [_decimal(value) for value in values]


def _file(path: Path, sop_class: str, key: str, series: str, study: str, date: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = _uid(key)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = _uid("fixture-writer")
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = _uid(study)
    dataset.SeriesInstanceUID = _uid(series)
    dataset.FrameOfReferenceUID = _uid(f"{study}-frame")
    dataset.PatientName = "Synthetic^Phantom"
    dataset.PatientID = "SYNTHETIC"
    dataset.StudyID = study
    dataset.StudyDate = date
    dataset.SeriesDate = date
    dataset.AcquisitionDate = date
    dataset.ContentDate = date
    dataset.StudyTime = "120000"
    dataset.SeriesTime = "120000"
    dataset.PatientPosition = "HFS"
    dataset.Manufacturer = "Synthetic fixture"
    return dataset


def _image_fields(dataset: Dataset, grid: _Grid, origin: NDArray[np.float64]) -> None:
    dataset.ImagePositionPatient = _decimals(origin)
    dataset.ImageOrientationPatient = _decimals(grid.direction[:, :2].T.ravel())
    dataset.PixelSpacing = _decimals(grid.spacing[:2][::-1])
    dataset.SliceThickness = _decimal(grid.spacing[2])
    dataset.Rows = grid.size[1]
    dataset.Columns = grid.size[0]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0


def _write(dataset: FileDataset, path: Path) -> None:
    if path.exists():
        buffer = BytesIO()
        pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
        if path.read_bytes() == buffer.getvalue():
            return
        raise FileExistsError(f"Select a new synthetic directory; an existing file differs: {path}")
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)


def _write_series(
    path: Path,
    grid: _Grid,
    values: NDArray[np.float64],
    *,
    study: str,
    date: str,
    label: str,
    description: str,
    modality: str = "MR",
    slope: float = 1.0,
    intercept: float = 0.0,
) -> str:
    path.mkdir(parents=True, exist_ok=True)
    sop_class = MRImageStorage if modality == "MR" else CTImageStorage
    pixels = np.rint((values - intercept) / slope).astype("<u2")
    # Reverse file order relative to patient geometry. Images must be sorted by
    # IPP projected on the slice normal to reconstruct the original 3D volume.
    for index, plane in enumerate(pixels):
        filename = path / f"slice_{grid.size[2] - index:03d}.dcm"
        series = f"{study}-{label}"
        dataset = _file(filename, sop_class, f"{series}-{index}", series, study, date)
        origin = grid.origin + index * grid.spacing[2] * grid.direction[:, 2]
        _image_fields(dataset, grid, origin)
        dataset.SeriesDescription = description
        dataset.Modality = modality
        dataset.SeriesNumber = {"anatomy": 1, "perfusion": 2, "ct": 3}[label]
        dataset.InstanceNumber = int(grid.size[2] - index)
        dataset.ImageType = ["DERIVED" if label == "perfusion" else "ORIGINAL", "PRIMARY", "OTHER"]
        dataset.SpacingBetweenSlices = _decimal(grid.spacing[2])
        dataset.RescaleSlope = _decimal(slope)
        dataset.RescaleIntercept = _decimal(intercept)
        if modality == "MR":
            dataset.ScanningSequence = "GR"
            dataset.SequenceVariant = "NONE"
            dataset.MRAcquisitionType = "3D" if label == "anatomy" else "2D"
            dataset.ProtocolName = description
            if label == "perfusion":
                # The phantom has no defined perfusion outside tissue or in CSF.
                dataset.PixelPaddingValue = 0
        else:
            dataset.RescaleType = "HU"
        dataset.PixelData = plane.tobytes()
        _write(dataset, filename)
    return str(path)


_DOSE_SCALE = 0.004


def _write_rtdose(path: Path, grid: _Grid, coordinates: NDArray[np.float64]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    filename = path / "accumulated_dose.dcm"
    dataset = _file(filename, RTDoseStorage, "planning-dose-0", "planning-dose", "planning", "19991130")
    _image_fields(dataset, grid, grid.origin)
    dataset.Modality = "RTDOSE"
    dataset.SeriesNumber = 4
    dataset.InstanceNumber = 1
    dataset.SeriesDescription = "Synthetic accumulated EQD2 demonstration"
    dataset.NumberOfFrames = grid.size[2]
    dataset.GridFrameOffsetVector = _decimals(np.arange(grid.size[2]) * grid.spacing[2])
    dataset.FrameIncrementPointer = Tag(0x3004, 0x000C)
    dataset.DoseUnits = "GY"
    dataset.DoseType = "EFFECTIVE"
    dataset.DoseSummationType = "MULTI_PLAN"
    dataset.DoseGridScaling = _decimal(_DOSE_SCALE)
    dataset.DoseComment = "Analytic fixture; no clinical dose calculation"
    dataset.PixelData = np.rint(_dose(coordinates) / _DOSE_SCALE).astype("<u2").tobytes()
    _write(dataset, filename)
    return str(path)


def _write_mask(path: Path, grid: _Grid, mask: NDArray[np.bool_]) -> str:
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.SetOrigin(grid.origin.tolist())
    image.SetSpacing(grid.spacing)
    image.SetDirection(grid.direction.ravel().tolist())
    if path.exists():
        existing = sitk.ReadImage(str(path))
        if (
            existing.IsSameImageGeometryAs(image, 1e-5, 1e-6)
            and np.array_equal(sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(existing))
        ):
            return str(path)
        raise FileExistsError(f"Select a new synthetic directory; an existing mask differs: {path}")
    sitk.WriteImage(image, str(path))
    return str(path)


def generate_synthetic_dataset(root: str | Path) -> dict:
    """Write synthetic DICOM and independent masks, then return pipeline configuration."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    center = np.array((37.4, -82.3, 21.6))
    may_pose = _Pose(center, (0, 0, 0), (0, 0, 0))
    ct_pose = _Pose(center, (3.2, -3.8, -4.2), (-3.7, 2.6, 3.0))
    october_pose = _Pose(center, (-2.5, 3.1, 4.8), (3.3, -2.6, 1.8))

    may_grid = _grid((52, 48, 42), (1.35, 1.40, 1.55), center, (3, -6, 8))
    october_grid = _grid((50, 49, 44), (1.43, 1.44, 1.54), center, (-4, -3, 3))
    ct_grid = _grid((58, 55, 49), (1.22, 1.29, 1.38), center, (3, 2, -6))
    may_cbv_grid = _grid((38, 35, 20), (1.75, 1.88, 3.15), center, (3, -6, 8))
    october_cbv_grid = _grid((39, 36, 18), (1.70, 1.83, 3.43), center + (0.6, 0.7, 1.2), (-4, -3, 3))
    dose_grid = _grid((36, 34, 28), (1.9, 2.0, 2.2), center + (-0.8, 0.5, 1.1), (-2, 4, 7))

    may_coordinates = may_pose.canonical(may_grid.coordinates())
    october_coordinates = october_pose.canonical(october_grid.coordinates())
    ct_coordinates = ct_pose.canonical(ct_grid.coordinates())
    may_cbv_coordinates = may_pose.canonical(may_cbv_grid.coordinates())
    october_cbv_coordinates = october_pose.canonical(october_cbv_grid.coordinates())
    dose_coordinates = ct_pose.canonical(dose_grid.coordinates())

    may_study = root / "may_study"
    october_study = root / "october_study"
    planning_study = root / "planning_study"
    paths = {
        "may_anatomy": _write_series(
            may_study / "anatomy", may_grid, _anatomy(may_coordinates, "MR"),
            study="may", date="20000517", label="anatomy", description="Synthetic 3D T1 MPRAGE",
        ),
        "october_anatomy": _write_series(
            october_study / "anatomy", october_grid, _anatomy(october_coordinates, "MR"),
            study="october", date="20001012", label="anatomy", description="Synthetic 3D T1 MPRAGE",
        ),
        "ct": _write_series(
            planning_study / "ct", ct_grid, _anatomy(ct_coordinates, "CT"),
            study="planning", date="19991130", label="ct", description="Synthetic planning CT",
            modality="CT", intercept=-1024.0,
        ),
        "may_rcbv": _write_series(
            may_study / "perfusion", may_cbv_grid, _perfusion(may_cbv_coordinates, False),
            study="may", date="20000517", label="perfusion",
            description="Synthetic leakage-corrected rCBV", slope=0.001,
        ),
        "october_rcbv": _write_series(
            october_study / "perfusion", october_cbv_grid, _perfusion(october_cbv_coordinates, True),
            study="october", date="20001012", label="perfusion",
            description="Synthetic leakage-corrected rCBV", slope=0.001,
        ),
        "dose": _write_rtdose(planning_study / "dose", dose_grid, dose_coordinates),
        "brain_mask": _write_mask(root / "may_brain.nii.gz", may_grid, _brain_mask(may_coordinates)),
        "october_brain_mask": _write_mask(
            root / "october_brain.nii.gz", october_grid, _brain_mask(october_coordinates)
        ),
        "ct_brain_mask": _write_mask(root / "ct_brain.nii.gz", ct_grid, _brain_mask(ct_coordinates)),
        "reference_roi": _write_mask(
            root / "may_reference_wm.nii.gz", may_grid,
            _ellipsoid(may_coordinates, _REFERENCE_CENTER, (4.0, 4.5, 4.8)) <= 1,
        ),
    }

    fixed_landmarks = center + np.array([
        [-13.5, 4.0, 9.0],
        [9.0, -13.0, 5.0],
        [7.0, 10.0, -4.0],
        [-2.5, 1.0, -15.0],
        [2.0, 3.0, 1.0],
        [12.0, 0.0, 15.0],
    ])
    poses = {"ct": ct_pose, "october": october_pose}
    landmarks = {
        name: {"fixed_lps": fixed_landmarks.tolist(), "moving_lps": pose.map_points(fixed_landmarks).tolist()}
        for name, pose in poses.items()
    }
    dose_index = (17, 16, 13)
    dose_lps = dose_grid.origin + dose_grid.direction @ (np.asarray(dose_index) * dose_grid.spacing)
    truth = {
        "fixed_to_moving": {name: pose.truth() for name, pose in poses.items()},
        "tre_tolerance_mm": 0.75,
        "may_rcbv_nominal_reference": 2.4,
        "october_rcbv_nominal_reference": 3.6,
        "may_reference_center_lps_mm": (center + _REFERENCE_CENTER).tolist(),
        "decrease_center_lps_mm": (center + _DECREASE_CENTER).tolist(),
        "increase_center_lps_mm": (center + _INCREASE_CENTER).tolist(),
        "dose_sample": {
            "ijk": list(dose_index),
            "lps_mm": dose_lps.tolist(),
            "value_gy": float(np.rint(_dose(dose_coordinates)[dose_index[::-1]] / _DOSE_SCALE) * _DOSE_SCALE),
            "dicom_scale": _DOSE_SCALE,
        },
    }

    truth_file = root / "ground_truth.json"
    truth_json = json.dumps(truth, indent=2) + "\n"
    if truth_file.exists():
        if truth_file.read_text(encoding="utf-8") != truth_json:
            raise FileExistsError(f"Select a new synthetic directory; ground truth differs: {truth_file}")
    else:
        truth_file.write_text(truth_json, encoding="utf-8")
    return {
        "mode": "synthetic",
        "paths": paths,
        "registration": {
            "same_frame_rcbv": {"may": True, "october": True},
            "metrics": {"ct": "mutual_information", "october": "correlation"},
            "options": {
                # Window the phantom CT for soft-tissue contrast before building
                # the pyramid. Keep the original values for resampling and QA.
                "ct": {"moving_clip": [0.0, 60.0]},
                "october": {"sampling_fraction": 1.0},
            },
        },
        "dose": {
            "interpretation": "eqd2",
            "evidence": "Known synthetic dose scale. The analytic fixture does not model clinical treatment delivery.",
        },
        "landmarks": landmarks,
        "synthetic_truth": truth,
    }
