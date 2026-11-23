"""Read quantitative DICOM volumes in DICOM patient coordinates (millimetres, LPS)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut
from pydicom.tag import Tag
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RTDoseStorage,
)

LOGGER = logging.getLogger(__name__)
POSITION_TOLERANCE_MM = 0.02
DIRECTION_TOLERANCE = 1e-4
OFFSET_ENCODING_TOLERANCE_MM = 1e-6


@dataclass(frozen=True)
class SeriesInfo:
    """DICOM objects grouped by study and series identifiers."""

    series_uid: str
    study_uid: str
    modality: str
    description: str
    study_date: str
    frame_uid: str | None
    paths: tuple[Path, ...]
    frame_count: int
    protocol: str = ""

    def to_dict(self) -> dict:
        """Return inspection metadata without source filenames or patient identifiers."""
        return {
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "modality": self.modality,
            "description": self.description,
            "study_date": self.study_date,
            "frame_uid": self.frame_uid,
            "object_count": len(self.paths),
            "frame_count": self.frame_count,
            "protocol": self.protocol,
        }


@dataclass
class LoadedImage:
    """A regular 3D volume and the evidence used to construct it."""

    image: sitk.Image
    frame_uid: str | None
    metadata: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "frame_uid": self.frame_uid,
            "geometry": {
                "coordinate_system": "DICOM LPS",
                "size_xyz": list(self.image.GetSize()),
                "spacing_mm_xyz": list(self.image.GetSpacing()),
                "origin_mm_lps": list(self.image.GetOrigin()),
                "direction_row_major": list(self.image.GetDirection()),
            },
            "warnings": list(self.warnings),
        }


def _uid(ds: Dataset, name: str) -> str:
    value = ds.get(name)
    if not value or not pydicom.uid.UID(str(value)).is_valid:
        raise ValueError(f"DICOM object requires a valid {name}.")
    return str(value)


def _read_header(path: Path) -> Dataset:
    try:
        return pydicom.dcmread(path, stop_before_pixels=True)
    except InvalidDicomError:
        # Raw data sets may lack a Part 10 preamble. Require recognizable DICOM
        # identities before considering them valid entries in the inventory.
        with path.open("rb") as stream:
            group = stream.read(2)
        if not any(int.from_bytes(group, byteorder) in {0x0002, 0x0008}
                   for byteorder in ("big", "little")):
            raise
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        for name in ("SOPClassUID", "SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID"):
            _uid(ds, name)
        return ds


def scan_series(directory: str | Path, *, recursive: bool = True) -> list[SeriesInfo]:
    """Inventory readable DICOM objects by StudyInstanceUID and SeriesInstanceUID."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM directory is missing or unreadable: {root}")

    def on_error(error: OSError) -> None:
        raise error

    groups: dict[tuple[str, str], list[tuple[Path, Dataset, int]]] = {}
    skipped = 0
    for folder, _, files in os.walk(root, onerror=on_error):
        for name in sorted(files):
            path = Path(folder) / name
            if not path.is_file() or name.upper() == "DICOMDIR":
                continue
            try:
                ds = _read_header(path)
                key = (_uid(ds, "StudyInstanceUID"), _uid(ds, "SeriesInstanceUID"))
                _uid(ds, "SOPClassUID")
                _uid(ds, "SOPInstanceUID")
                if not ds.get("Modality"):
                    raise ValueError("Missing Modality.")
                count = _positive_int(ds, "NumberOfFrames", default=1)
            except (InvalidDicomError, ValueError, EOFError, TypeError):
                skipped += 1
                continue
            groups.setdefault(key, []).append((path, ds, count))
        if not recursive:
            break
    if skipped:
        LOGGER.warning("DICOM inventory skipped %d invalid or non-series files.", skipped)

    inventory = []
    for (study, series), items in sorted(groups.items()):
        ds = items[0][1]
        frames = {str(item.get("FrameOfReferenceUID", "")) for _, item, _ in items}
        count = sum(number for _, _, number in items)
        inventory.append(
            SeriesInfo(
                series_uid=series,
                study_uid=study,
                modality=str(ds.Modality),
                description=str(ds.get("SeriesDescription", "")),
                study_date=str(ds.get("StudyDate", "")),
                frame_uid=next(iter(frames)) if len(frames) == 1 and "" not in frames else None,
                paths=tuple(path for path, _, _ in items),
                frame_count=count,
                protocol=str(ds.get("ProtocolName", "")),
            )
        )
    return inventory


def find_anatomical_candidates(
    study_directory: str | Path,
    *,
    study_uid: str | None = None,
    excluded_series_uid: str | None = None,
) -> list[SeriesInfo]:
    """List potential T1, T2 and FLAIR volumes for review in one MRI study."""
    candidates = []
    excluded_terms = (
        "rcbv", "cbv", "perfusion", "scout", "localizer", "adc", "dwi", "subtraction"
    )
    anatomical_terms = ("t1", "t2", "flair", "mprage", "spgr", "bravo", "enhanc")
    for series in scan_series(study_directory):
        label = f"{series.description} {series.protocol}".lower()
        if (
            series.modality != "MR"
            or series.series_uid == excluded_series_uid
            or (study_uid is not None and series.study_uid != study_uid)
            or series.frame_count < 2
            or any(term in label for term in excluded_terms)
        ):
            continue
        recognized = any(term in label for term in anatomical_terms)
        candidates.append((recognized, series))
    candidates.sort(key=lambda item: (not item[0], -item[1].frame_count, item[1].series_uid))
    return [item[1] for item in candidates]


def _select_series(
    directory: str | Path, series_uid: str | None, modality: str | None
) -> SeriesInfo:
    inventory = scan_series(directory)
    matching = [
        entry for entry in inventory
        if (series_uid is None or series_uid == entry.series_uid)
        and (modality is None or modality == entry.modality)
    ]
    if not matching:
        raise ValueError(f"No matching DICOM series for modality {modality or 'any'}.")
    if len(matching) != 1:
        choices = ", ".join(f"{s.series_uid} ({s.modality}, {s.frame_count} frames)"
                            for s in matching)
        raise ValueError(f"Select a DICOM SeriesInstanceUID; multiple series match: {choices}")
    return matching[0]


def _read_objects(series: SeriesInfo) -> list[Dataset]:
    objects = [pydicom.dcmread(path, force=True) for path in series.paths]
    sop_uids = [_uid(ds, "SOPInstanceUID") for ds in objects]
    if len(set(sop_uids)) != len(sop_uids):
        raise ValueError("Duplicate SOPInstanceUIDs in selected DICOM series.")
    if {str(ds.Modality) for ds in objects} != {series.modality}:
        raise ValueError("Inconsistent Modality within selected DICOM series.")
    if any(
        str(ds.get("SeriesInstanceUID", "")) != series.series_uid
        or str(ds.get("StudyInstanceUID", "")) != series.study_uid
        for ds in objects
    ):
        raise ValueError("DICOM study or series identity changed after inspection.")
    if any(str(ds.get("AnatomicalOrientationType") or "BIPED") != "BIPED" for ds in objects):
        raise ValueError("This reader supports human (BIPED) patient-coordinate orientation.")
    return objects


def _frame_uid(objects: list[Dataset], warnings: list[str]) -> str | None:
    values = {str(ds.get("FrameOfReferenceUID", "")) for ds in objects}
    if len(values) != 1:
        raise ValueError("Inconsistent FrameOfReferenceUID within selected DICOM images.")
    frame = values.pop()
    if not frame:
        warnings.append("Missing DICOM FrameOfReferenceUID; spatial relationships require external evidence.")
        return None
    return _uid(objects[0], "FrameOfReferenceUID")


@dataclass(frozen=True)
class _Frame:
    pixels: np.ndarray
    dataset: Dataset
    per_frame: Dataset | None
    shared: Dataset | None

    def get(self, name: str, group_name: str | None = None, default=None):
        for group in (self.per_frame, self.shared):
            if group is None:
                continue
            if group_name is not None:
                sequence = group.get(group_name)
                if sequence is not None:
                    if len(sequence) != 1:
                        raise ValueError(f"[DICOM] {group_name} must contain one item per frame.")
                    if name in sequence[0]:
                        return sequence[0].get(name)
            elif name in group:
                return group.get(name)
        return self.dataset.get(name, default)


def _decode_pixels(ds: Dataset) -> np.ndarray:
    if _positive_int(ds, "SamplesPerPixel") != 1:
        raise ValueError("Quantitative analysis requires a single-channel DICOM image.")
    if not ds.file_meta.get("TransferSyntaxUID"):
        # Native data sets without Part 10 metadata have an inferable encoding.
        # Encapsulated pixel data require the original transfer syntax.
        implicit, little_endian = ds.original_encoding
        pixel_data = next(
            (ds[tag] for tag in (0x7FE00008, 0x7FE00009, 0x7FE00010) if tag in ds), None
        )
        if pixel_data is not None and not pixel_data.is_undefined_length:
            if implicit and little_endian:
                ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            elif not implicit:
                ds.file_meta.TransferSyntaxUID = (
                    ExplicitVRLittleEndian if little_endian else ExplicitVRBigEndian
                )
    try:
        return np.asarray(ds.pixel_array)
    except (AttributeError, RuntimeError, ValueError, NotImplementedError) as exc:
        transfer = ds.file_meta.get("TransferSyntaxUID") if hasattr(ds, "file_meta") else None
        raise ValueError(
            f"Unable to decode DICOM pixels (transfer syntax {transfer or 'unspecified'}). "
            "Check file integrity and install the pydicom decoder for that transfer syntax if needed."
        ) from exc


def _frames(ds: Dataset) -> Iterable[_Frame]:
    pixels = _decode_pixels(ds)
    rows, columns = _positive_int(ds, "Rows"), _positive_int(ds, "Columns")
    number = _positive_int(ds, "NumberOfFrames", default=1)
    if pixels.shape != ((rows, columns) if number == 1 else (number, rows, columns)):
        raise ValueError("DICOM image dimensions disagree with the decoded pixel array.")
    per_frame = ds.get("PerFrameFunctionalGroupsSequence")
    shared_items = ds.get("SharedFunctionalGroupsSequence")
    if per_frame is not None and len(per_frame) != number:
        raise ValueError("Per-frame functional groups disagree with NumberOfFrames.")
    if shared_items is not None and len(shared_items) != 1:
        raise ValueError("SharedFunctionalGroupsSequence must contain one item.")
    if number > 1 and per_frame is None:
        raise ValueError("Multiframe DICOM image lacks per-frame geometry.")
    shared = shared_items[0] if shared_items else None
    for index, plane in enumerate(pixels[np.newaxis, ...] if number == 1 else pixels):
        yield _Frame(plane, ds, per_frame[index] if per_frame else None, shared)


def _vector(value, count: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {count} finite numeric values.") from exc
    if vector.shape != (count,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {count} finite numeric values.")
    return vector


def _scalar(value, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {name}.") from exc
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Missing or invalid {name}.")
    return number


def _positive_int(ds: Dataset, name: str, *, default: int | None = None) -> int:
    value = _scalar(ds.get(name, default), name, positive=True)
    if not value.is_integer():
        raise ValueError(f"DICOM {name} must be a positive integer.")
    return int(value)


def _direction(orientation) -> np.ndarray:
    # IOP supplies the direction along a row (column index) and along a
    # column (row index). The cross product defines the slice normal.
    first, second = _vector(orientation, 6, "ImageOrientationPatient").reshape(2, 3)
    if (
        abs(np.linalg.norm(first) - 1) > DIRECTION_TOLERANCE
        or abs(np.linalg.norm(second) - 1) > DIRECTION_TOLERANCE
        or abs(first @ second) > DIRECTION_TOLERANCE
    ):
        raise ValueError("Invalid or nonorthogonal DICOM ImageOrientationPatient.")
    first = first / np.linalg.norm(first)
    second = second - (first @ second) * first
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def _pixel_spacing(value) -> np.ndarray:
    spacing = _vector(value, 2, "PixelSpacing")
    if (spacing <= 0).any():
        raise ValueError("DICOM PixelSpacing must be positive for a 3D image.")
    return spacing[::-1]


def _spacing_between_positions(positions: np.ndarray, label: str) -> float:
    distances = np.diff(positions)
    if (distances <= POSITION_TOLERANCE_MM).any():
        raise ValueError(f"{label} has duplicated or indistinguishable slice positions.")
    spacing = float(np.mean(distances))
    expected = positions[0] + spacing * np.arange(positions.size)
    if not np.allclose(positions, expected, atol=POSITION_TOLERANCE_MM, rtol=0):
        raise ValueError(f"{label} has nonuniform slice spacing or missing slices.")
    return spacing


def _sitk_image(
    array: np.ndarray, origin: np.ndarray, spacing: Iterable[float], direction: np.ndarray
) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    image.SetOrigin(origin.tolist())
    image.SetSpacing(tuple(spacing))
    image.SetDirection(direction.ravel().tolist())
    return image


def _stored_validity(frame: _Frame) -> np.ndarray:
    pixels = frame.pixels
    ds = frame.dataset
    valid = np.isfinite(pixels)
    for value_name, range_name in (
        ("PixelPaddingValue", "PixelPaddingRangeLimit"),
        ("FloatPixelPaddingValue", "FloatPixelPaddingRangeLimit"),
        ("DoubleFloatPixelPaddingValue", "DoubleFloatPixelPaddingRangeLimit"),
    ):
        if value_name in ds:
            first = ds.get(value_name)
            last = ds.get(range_name, first)
            valid &= (pixels < min(first, last)) | (pixels > max(first, last))
    return valid


def _rwvm(values: np.ndarray, valid: np.ndarray, sequence) -> tuple[np.ndarray, dict]:
    if len(sequence) != 1:
        raise ValueError("Multiple DICOM real-world mappings require an explicit scaling selection.")
    mapping = sequence[0]
    first = _scalar(
        mapping.get("DoubleFloatRealWorldValueFirstValueMapped",
                    mapping.get("RealWorldValueFirstValueMapped")),
        "RealWorldValueFirstValueMapped",
    )
    last = _scalar(
        mapping.get("DoubleFloatRealWorldValueLastValueMapped",
                    mapping.get("RealWorldValueLastValueMapped")),
        "RealWorldValueLastValueMapped",
    )
    if first > last:
        raise ValueError("Invalid DICOM real-world mapping range.")
    valid &= (values >= first) & (values <= last)
    record = {"method": "real_world", "stored_range": [first, last]}
    units = mapping.get("MeasurementUnitsCodeSequence", [])
    if units:
        record["units"] = {
            "code": str(units[0].get("CodeValue", "")),
            "scheme": str(units[0].get("CodingSchemeDesignator", "")),
            "meaning": str(units[0].get("CodeMeaning", "")),
        }
    lut = mapping.get("RealWorldValueLUTData")
    if lut is not None:
        lut = np.atleast_1d(np.asarray(lut, dtype=float))
        if (
            first != int(first)
            or last != int(last)
            or lut.shape != (int(last - first) + 1,)
            or not np.isfinite(lut).all()
        ):
            raise ValueError("DICOM real-world lookup table disagrees with its stored range.")
        if np.any(values[valid] % 1):
            raise ValueError("DICOM real-world lookup table requires integer pixel values.")
        mapped = np.full(values.shape, np.nan, dtype=float)
        mapped[valid] = lut[values[valid].astype(np.int64) - int(first)]
        record["lut_entries"] = int(lut.size)
        return mapped, record
    slope = _scalar(mapping.get("RealWorldValueSlope"), "RealWorldValueSlope")
    intercept = _scalar(mapping.get("RealWorldValueIntercept"), "RealWorldValueIntercept")
    if slope == 0:
        raise ValueError("DICOM RealWorldValueSlope cannot be zero.")
    record.update(slope=slope, intercept=intercept)
    return values * slope + intercept, record


def _scale_frame(
    frame: _Frame, scaling: str, scale_override: tuple[float, float] | None
) -> tuple[np.ndarray, dict]:
    values = frame.pixels.astype(np.float64)
    valid = _stored_validity(frame)
    rwvm = frame.get("RealWorldValueMappingSequence")
    slope = frame.get("RescaleSlope", "PixelValueTransformationSequence")
    intercept = frame.get("RescaleIntercept", "PixelValueTransformationSequence")
    if scale_override is not None:
        slope, intercept = _vector(scale_override, 2, "vendor scaling slope and intercept")
        if slope == 0:
            raise ValueError("Configured vendor scaling slope cannot be zero.")
        values = values * slope + intercept
        record = {"method": "vendor_override", "slope": float(slope), "intercept": float(intercept)}
    elif scaling == "stored":
        record = {"method": "stored"}
    elif scaling == "real_world" or (scaling == "auto" and rwvm is not None):
        if not rwvm:
            raise ValueError("Requested real-world scaling requires RealWorldValueMappingSequence.")
        values, record = _rwvm(values, valid, rwvm)
    elif frame.dataset.get("ModalityLUTSequence"):
        values = np.asarray(apply_modality_lut(frame.pixels, frame.dataset), dtype=np.float64)
        record = {"method": "modality_lut"}
    elif slope is not None or intercept is not None:
        slope = _scalar(slope, "RescaleSlope")
        intercept = _scalar(intercept, "RescaleIntercept")
        if slope == 0:
            raise ValueError("DICOM RescaleSlope cannot be zero.")
        values = values * slope + intercept
        record = {"method": "modality_rescale", "slope": slope, "intercept": intercept,
                  "rescale_type": str(frame.get("RescaleType", "PixelValueTransformationSequence", ""))}
    else:
        record = {"method": "stored"}
    values[~valid] = np.nan
    if not np.isfinite(values[valid]).all() or np.any(np.abs(values[valid]) > np.finfo(np.float32).max):
        raise ValueError("DICOM quantitative scaling exceeds the supported numeric range.")
    return values.astype(np.float32), record


def _metadata(series: SeriesInfo, objects: list[Dataset]) -> dict:
    ds = objects[0]
    image_type = ds.get("ImageType") or []
    return {
        **series.to_dict(),
        "sop_instance_uids": [str(item.SOPInstanceUID) for item in objects],
        "sop_class_uids": sorted({str(item.SOPClassUID) for item in objects}),
        "acquisition_date": str(ds.get("AcquisitionDate", ds.get("AcquisitionDateTime", "")[:8])),
        "series_date": str(ds.get("SeriesDate", "")),
        "manufacturer": str(ds.get("Manufacturer", "")),
        "manufacturer_model": str(ds.get("ManufacturerModelName", "")),
        "image_type": [image_type] if isinstance(image_type, str) else [str(value) for value in image_type],
        "patient_position": str(ds.get("PatientPosition", "")),
        "referenced_series_uids": sorted({
            str(item.SeriesInstanceUID)
            for obj in objects
            for item in obj.get("ReferencedSeriesSequence", [])
            if item.get("SeriesInstanceUID")
        }),
        "source_image_uids": sorted({
            str(item.ReferencedSOPInstanceUID)
            for obj in objects
            for item in obj.get("SourceImageSequence", [])
            if item.get("ReferencedSOPInstanceUID")
        }),
    }


def read_volume(
    directory: str | Path,
    *,
    series_uid: str | None = None,
    modality: str | None = None,
    scaling: str = "auto",
    scale_override: tuple[float, float] | None = None,
) -> LoadedImage:
    """Read a single MR or CT volume, ordering planes using their physical centers.

    Use `scale_override` only for a documented vendor mapping applied directly
    to stored pixels. Real-world mapping and modality scaling each operate on
    stored pixels and are never stacked. Invalid/padding samples are NaN.
    """
    if scaling not in {"auto", "stored", "modality", "real_world"}:
        raise ValueError(f"Unknown DICOM image scaling method {scaling!r}.")
    if scale_override is not None and scaling != "auto":
        raise ValueError("Configure a vendor scale override or a named scaling method.")
    series = _select_series(directory, series_uid, modality)
    if series.modality not in {"MR", "CT"}:
        raise ValueError(f"Expected an MR or CT image series, found {series.modality}.")
    objects = _read_objects(series)
    warnings: list[str] = []
    frame_uid = _frame_uid(objects, warnings)
    frames = [frame for ds in objects for frame in _frames(ds)]
    first = frames[0]
    direction = _direction(first.get("ImageOrientationPatient", "PlaneOrientationSequence"))
    spacing_xy = _pixel_spacing(first.get("PixelSpacing", "PixelMeasuresSequence"))
    positions = []
    reported_steps = []
    thicknesses = []
    for frame in frames:
        if frame.pixels.shape != first.pixels.shape:
            raise ValueError("DICOM series has inconsistent Rows or Columns.")
        orientation = _direction(frame.get("ImageOrientationPatient", "PlaneOrientationSequence"))
        if not np.allclose(orientation, direction, atol=DIRECTION_TOLERANCE, rtol=0):
            raise ValueError("DICOM series has inconsistent image orientations.")
        frame_spacing = _pixel_spacing(frame.get("PixelSpacing", "PixelMeasuresSequence"))
        if not np.allclose(frame_spacing, spacing_xy, atol=DIRECTION_TOLERANCE, rtol=0):
            raise ValueError("DICOM series has inconsistent PixelSpacing.")
        positions.append(_vector(frame.get("ImagePositionPatient", "PlanePositionSequence"),
                                 3, "ImagePositionPatient"))
        reported_step = frame.get("SpacingBetweenSlices", "PixelMeasuresSequence")
        if reported_step is not None:
            reported_steps.append(abs(_scalar(reported_step, "SpacingBetweenSlices")))
        thickness = frame.get("SliceThickness", "PixelMeasuresSequence")
        if thickness is not None:
            thicknesses.append(_scalar(thickness, "SliceThickness", positive=True))

    positions = np.asarray(positions)
    coordinates = positions @ direction
    order = np.argsort(coordinates[:, 2])
    if not np.allclose(coordinates[:, :2], coordinates[order[0], :2],
                       atol=POSITION_TOLERANCE_MM, rtol=0):
        raise ValueError("DICOM slices have in-plane offsets; a tilted or sheared grid requires reconstruction.")
    if len(frames) > 1:
        spacing_z = _spacing_between_positions(coordinates[order, 2], "DICOM image series")
        if reported_steps and not np.allclose(
            reported_steps, spacing_z, atol=POSITION_TOLERANCE_MM, rtol=0
        ):
            if max(reported_steps) < spacing_z - POSITION_TOLERANCE_MM:
                raise ValueError("DICOM slice positions leave gaps relative to SpacingBetweenSlices.")
            warnings.append("DICOM SpacingBetweenSlices disagrees with spacing measured from slice positions.")
    elif reported_steps or thicknesses:
        spacing_z = _scalar((reported_steps or thicknesses)[0], "single-slice spacing", positive=True)
        warnings.append("The series contains one slice; through-plane coverage is limited.")
    else:
        raise ValueError("A single-slice image requires SliceThickness or SpacingBetweenSlices.")

    planes = np.empty((len(frames), *first.pixels.shape), dtype=np.float32)
    applied = []
    scale_indices = []
    for output_index, index in enumerate(order):
        values, record = _scale_frame(frames[index], scaling, scale_override)
        planes[output_index] = values
        if record not in applied:
            applied.append(record)
        scale_indices.append(applied.index(record))
    if series.modality == "MR" and {item["method"] for item in applied} == {"stored"}:
        warnings.append("MR pixels are stored values; verify vendor scaling and quantitative interpretation.")
    if len({item["method"] for item in applied}) != 1:
        raise ValueError("DICOM scaling methods are inconsistent between slices.")
    units = {
        (item.get("rescale_type", ""), item.get("units", {}).get("code", ""),
         item.get("units", {}).get("scheme", ""))
        for item in applied
    }
    if len(units) != 1:
        raise ValueError("DICOM value scaling units are inconsistent between slices.")

    metadata = _metadata(series, objects)
    metadata.update(
        image_orientation_patient=list(direction[:, 0]) + list(direction[:, 1]),
        image_positions_patient_lps_mm=positions[order].tolist(),
        rescale_slope_intercept=[
            list(pair)
            for pair in sorted({
                (str(frame.get("RescaleSlope", "PixelValueTransformationSequence", "")),
                 str(frame.get("RescaleIntercept", "PixelValueTransformationSequence", "")))
                for frame in frames
            })
        ],
        slice_thickness_mm=sorted(set(thicknesses)),
        real_world_mapping_present=any(frame.get("RealWorldValueMappingSequence") for frame in frames),
        value_scaling=applied,
        value_scaling_index_by_slice=scale_indices,
        valid_voxel_count=int(np.count_nonzero(np.isfinite(planes))),
    )
    return LoadedImage(
        _sitk_image(planes, positions[order[0]], (*spacing_xy, spacing_z), direction),
        frame_uid,
        metadata,
        warnings,
    )


def _referenced(ds: Dataset, sequence: str) -> list[dict]:
    return [
        {
            "sop_instance_uid": str(item.get("ReferencedSOPInstanceUID", "")),
            "sop_class_uid": str(item.get("ReferencedSOPClassUID", "")),
        }
        for item in ds.get(sequence, [])
    ]


def read_rtdose(
    directory: str | Path, *, series_uid: str | None = None, sop_uid: str | None = None
) -> LoadedImage:
    """Read one RTDOSE object, applying DoseGridScaling once in its declared units."""
    series = _select_series(directory, series_uid, "RTDOSE")
    objects = _read_objects(series)
    selected = [ds for ds in objects if sop_uid is None or str(ds.SOPInstanceUID) == sop_uid]
    if len(selected) != 1:
        raise ValueError("Select one RTDOSE object using its SOPInstanceUID.")
    ds = selected[0]
    if str(ds.SOPClassUID) != RTDoseStorage:
        raise ValueError("The selected DICOM object is not an RT Dose Storage object.")
    warnings: list[str] = []
    frame_uid = _frame_uid(selected, warnings)
    scale = _scalar(ds.get("DoseGridScaling"), "DoseGridScaling", positive=True)
    units = str(ds.get("DoseUnits", "")).strip().upper()
    dose_type = str(ds.get("DoseType", "")).strip().upper()
    if units not in {"GY", "RELATIVE"}:
        raise ValueError(f"Missing or unsupported DICOM DoseUnits: {units!r}.")
    if dose_type not in {"PHYSICAL", "EFFECTIVE", "ERROR"}:
        raise ValueError(f"Missing or unsupported DICOM DoseType: {dose_type!r}.")
    if not ds.get("DoseSummationType"):
        warnings.append("Missing DoseSummationType; verify which treatments the dose represents.")

    ipp = _vector(ds.get("ImagePositionPatient"), 3, "ImagePositionPatient")
    direction = _direction(ds.get("ImageOrientationPatient"))
    spacing_xy = _pixel_spacing(ds.get("PixelSpacing"))
    number = _positive_int(ds, "NumberOfFrames", default=1)
    pixels = _decode_pixels(ds)
    rows, columns = _positive_int(ds, "Rows"), _positive_int(ds, "Columns")
    if number == 1 and pixels.shape == (rows, columns):
        pixels = pixels[np.newaxis, ...]
    if pixels.shape != (number, rows, columns):
        raise ValueError("RTDOSE image dimensions disagree with NumberOfFrames or decoded pixels.")

    raw_offsets = ds.get("GridFrameOffsetVector")
    encoding = "single_frame"
    if raw_offsets is None and number == 1:
        offsets = np.zeros(1)
    else:
        offsets = _vector(np.atleast_1d(raw_offsets), number, "GridFrameOffsetVector")
        if abs(offsets[0]) <= OFFSET_ENCODING_TOLERANCE_MM:
            encoding = "relative_to_first_frame"
        elif (
            np.allclose(direction, np.eye(3), atol=DIRECTION_TOLERANCE, rtol=0)
            and abs(offsets[0] - ipp[2]) <= OFFSET_ENCODING_TOLERANCE_MM
        ):
            encoding = "absolute_axial_patient_z"
            offsets = offsets - ipp[2]
        else:
            raise ValueError("Unsupported RTDOSE GridFrameOffsetVector encoding.")
    if number > 1:
        pointer = ds.get("FrameIncrementPointer")
        pointers = [] if pointer is None else list(np.atleast_1d(pointer))
        if Tag(0x3004000C) not in pointers:
            warnings.append("FrameIncrementPointer omits GridFrameOffsetVector; inspect dose frame geometry.")
        differences = np.diff(offsets)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError("RTDOSE GridFrameOffsetVector must be strictly monotonic.")
        order = np.argsort(offsets)
        spacing_z = _spacing_between_positions(offsets[order], "RTDOSE grid")
    else:
        order = np.array([0])
        thickness = ds.get("SliceThickness")
        spacing_z = _scalar(thickness, "SliceThickness", positive=True) if thickness is not None else 1.0
        warnings.append(
            "RTDOSE has one frame; through-plane extent is undefined by frame centers. "
            "Spacing uses SliceThickness when present, otherwise a 1 mm storage convention."
        )
    if (pixels < 0).any() and dose_type != "ERROR":
        raise ValueError("DICOM dose grid contains negative dose values.")
    dose = pixels[order].astype(np.float64) * scale
    if not np.isfinite(dose).all() or np.any(np.abs(dose) > np.finfo(np.float32).max):
        raise ValueError("DICOM dose grid exceeds the supported numeric range.")
    origin = ipp + offsets[order[0]] * direction[:, 2]
    plans = _referenced(ds, "ReferencedRTPlanSequence")
    if not plans:
        warnings.append("RTDOSE has no ReferencedRTPlanSequence; verify dose provenance separately.")
    metadata = _metadata(series, selected)
    metadata.update(
        description=str(ds.get("SeriesDescription", "")),
        protocol=str(ds.get("ProtocolName", "")),
        frame_count=number,
        object_count=1,
        dose_units=units,
        dose_type=dose_type,
        dose_grid_scaling=scale,
        dose_summation_type=str(ds.get("DoseSummationType", "")),
        grid_frame_offset_vector_mm=(
            None if raw_offsets is None else [float(value) for value in np.atleast_1d(raw_offsets)]
        ),
        grid_frame_offset_encoding=encoding,
        referenced_rt_plans=plans,
        referenced_spatial_registrations=_referenced(ds, "ReferencedSpatialRegistrationSequence"),
        spatial_transform_of_dose=str(ds.get("SpatialTransformOfDose", "")),
        derivation_text={
            name: str(ds.get(name, "")) for name in
            ("SeriesDescription", "DoseComment", "DerivationDescription")
            if ds.get(name)
        },
    )
    return LoadedImage(
        _sitk_image(dose, origin, (*spacing_xy, spacing_z), direction),
        frame_uid,
        metadata,
        warnings,
    )
