"""Shared stages used by the command line and Jupyter notebook."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from .brain_segmentation import analysis_masks, load_mask
from .dicom_io import find_anatomical_candidates, read_rtdose, read_volume
from .dose_analysis import analyze_dose
from .image_geometry import (
    array_to_image as array_image, compose_transforms, geometry_summary, resample_mask, resample_scalar,
)
from .normalization import normalize, summarize
from .rcbv_analysis import candidate_regions, candidates, change_maps, threshold_sensitivity
from .registration import register_rigid
from .registration_qa import evaluate_registration
from .reporting import serializable, write_csv, write_json, write_report

DEFAULTS = {
    "mode": "patient", "paths": {}, "series_uids": {},
    "rcbv": {"scaling": {}, "scale_override": {}, "quantitative_evidence": None, "exclude_negative": False},
    "registration": {
        "same_frame_rcbv": {}, "rcbv_method": {}, "options": {}, "thresholds": {},
        "review": {"status": "pending"}, "metrics": {},
    },
    "dose": {"interpretation": "unverified"},
    "normalization": {"method": "white_matter", "min_roi_voxels": 20},
    "analysis": {
        "baseline_floor": 0.05, "absolute": 0.25, "percent": None,
        "robust_sigma": None, "min_region_ml": 0, "sensitivity": [0.15, 0.25, 0.4, 0.6],
        "dose_thresholds": list(range(20, 81, 10)),
        "dose_bin_edges": list(range(0, 101, 10)), "overlay_dose_threshold": 40,
    },
}


@dataclass
class Inputs:
    images: dict[str, sitk.Image]
    loaded: dict
    report: dict


@dataclass
class Registered:
    images: dict[str, sitk.Image]
    coverage: dict[str, sitk.Image]
    transforms: dict[str, sitk.Transform]
    report: dict
    figures: list[Path]

    @property
    def reference(self):
        return self.images["may_anatomy"]


@dataclass
class Analysis:
    maps: dict[str, np.ndarray]
    candidate: np.ndarray
    valid: np.ndarray
    roi: np.ndarray
    dose_valid: np.ndarray
    spatial_overlap: np.ndarray
    report: dict


def _merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(path=None, *, overrides=None) -> dict:
    """Read YAML with paths relative to that YAML file when using relative paths."""
    config = deepcopy(DEFAULTS)
    if path is not None:
        path = Path(path).resolve()
        updates = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(updates, dict):
            raise ValueError("The configuration file must contain a YAML mapping.")
        for key, value in updates.get("paths", {}).items():
            if value and not Path(value).is_absolute() and value[1:2] != ":":
                updates["paths"][key] = str((path.parent / value).resolve())
        transform = updates.get("dose", {}).get("ct_to_dose_transform")
        if transform and not Path(transform).is_absolute() and transform[1:2] != ":":
            updates["dose"]["ct_to_dose_transform"] = str((path.parent / transform).resolve())
        _merge(config, updates)
    if overrides:
        _merge(config, overrides)
    if config["mode"] not in {"patient", "synthetic"}:
        raise ValueError("mode must be patient or synthetic.")
    return config


def _output_directory(config, output_dir):
    output = Path(output_dir).resolve()
    for value in config["paths"].values():
        if value and output.is_relative_to(Path(value).resolve()):
            raise ValueError(f"Output directory must be separate from input: {value}")
    for name in ("qa", "registered_images", "figures", "quantitative_results", "reports"):
        folder = output / name
        for value in config["paths"].values():
            if value and Path(value).resolve().is_relative_to(folder):
                raise ValueError(f"Input must be separate from generated files in {folder}: {value}")
        if folder.is_symlink() or (folder.is_dir() and any(child.is_symlink() for child in folder.iterdir())):
            raise ValueError(f"Output path includes a symbolic link under {folder}.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _anatomy(config, perfusion, session, report):
    paths = config["paths"]
    name = f"{session}_anatomy"
    selected_uid = config["series_uids"].get(name)
    if paths.get(name):
        anatomy = read_volume(paths[name], series_uid=selected_uid, modality="MR")
        reason = "Anatomy directory explicitly configured for this examination."
    else:
        study = paths.get(f"{session}_study") or Path(paths[f"{session}_rcbv"]).parent
        candidates_found = find_anatomical_candidates(
            study, study_uid=perfusion.metadata.get("study_uid"),
            excluded_series_uid=perfusion.metadata.get("series_uid"),
        )
        report["anatomy_candidates"][session] = [candidate.to_dict() for candidate in candidates_found]
        if selected_uid:
            choices = [candidate for candidate in candidates_found if candidate.series_uid == selected_uid]
        else:
            choices = candidates_found
        if len(choices) != 1:
            raise ValueError(
                f"{session}: found {len(candidates_found)} plausible anatomical series. "
                f"Review inspection.json and set paths.{name} or series_uids.{name}."
            )
        anatomy = read_volume(study, series_uid=choices[0].series_uid, modality="MR")
        reason = "Only matching candidate in the same DICOM study." if not selected_uid else (
            "Series explicitly selected from discovered anatomical candidates."
        )
    if anatomy.metadata.get("study_uid") != perfusion.metadata.get("study_uid"):
        raise ValueError(f"{session}: selected anatomy and rCBV belong to different DICOM studies.")
    if anatomy.metadata.get("series_uid") == perfusion.metadata.get("series_uid"):
        raise ValueError(f"{session}: selected anatomy is the rCBV series.")
    report["anatomy_selection"][session] = config.get("anatomy_selection_reason", {}).get(session) or reason
    return anatomy


def inspect_inputs(config, output_dir) -> Inputs:
    """Load images and write metadata and candidate series before any registration."""
    output = _output_directory(config, output_dir)
    loaded = {}
    report = {"images": {}, "anatomy_candidates": {}, "anatomy_selection": {}, "errors": []}
    paths = config["paths"]
    uids = config["series_uids"]
    for key, modality in (("may_rcbv", "MR"), ("october_rcbv", "MR"), ("ct", "CT"), ("dose", "RTDOSE")):
        try:
            if not paths.get(key) or not Path(paths[key]).exists():
                raise FileNotFoundError(f"{key} directory is unavailable: {paths.get(key)}")
            if modality == "RTDOSE":
                loaded[key] = read_rtdose(paths[key], series_uid=uids.get(key), sop_uid=uids.get("dose_sop"))
            else:
                session = key.removesuffix("_rcbv")
                if key.endswith("_rcbv"):
                    loaded[key] = read_volume(
                        paths[key], series_uid=uids.get(key), modality=modality,
                        scaling=config["rcbv"]["scaling"].get(session, "auto"),
                        scale_override=config["rcbv"]["scale_override"].get(session),
                    )
                else:
                    loaded[key] = read_volume(paths[key], series_uid=uids.get(key), modality=modality)
        except (OSError, ValueError, RuntimeError) as exc:
            report["errors"].append(str(exc))
    for session in ("may", "october"):
        if f"{session}_rcbv" in loaded:
            try:
                loaded[f"{session}_anatomy"] = _anatomy(
                    config, loaded[f"{session}_rcbv"], session, report,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                report["errors"].append(str(exc))
    for name, item in loaded.items():
        report["images"][name] = {
            **item.to_dict(),
            "pipeline_geometry": geometry_summary(item.image),
            "pipeline_date": item.metadata.get("acquisition_date") or item.metadata.get("study_date"),
        }
    write_json(output / "quantitative_results" / "inspection.json", report)
    if report["errors"]:
        raise ValueError("Input inspection failed:\n" + "\n".join(report["errors"]))
    return Inputs({name: item.image for name, item in loaded.items()}, loaded, report)


def _landmarks(config, name):
    points = config.get("landmarks", {}).get(name)
    if points is None:
        return None
    return {
        "fixed_lps": points.get("fixed_lps", points.get("fixed")),
        "moving_lps": points.get("moving_lps", points.get("moving")),
    }


def _native_mask(config, key, image):
    path = config["paths"].get(key)
    return array_image(load_mask(path, image).astype(np.uint8), image) if path else None


def _dose_chain(config, inputs):
    specification = config["dose"]
    info = inputs.loaded["dose"].metadata
    if info.get("dose_type") == "ERROR":
        raise ValueError("A DICOM DoseType ERROR grid cannot be analyzed as delivered radiation dose.")
    if inputs.images["dose"].GetSize()[2] < 2:
        raise ValueError("Volumetric dose analysis requires multiple distinct RTDOSE frames.")
    units = str(info.get("dose_units", "")).upper()
    if units == "GY":
        label = "RTDOSE grid (Gy, derivation unverified)"
    elif units == "RELATIVE":
        label = "Relative RTDOSE grid"
    else:
        raise ValueError(f"Unsupported RTDOSE units: {units!r}")
    interpretation = specification.get("interpretation", "unverified")
    evidence = specification.get("evidence")
    if interpretation == "eqd2" and units == "GY" and evidence:
        label = "Accumulated EQD2 (Gy)"
    elif interpretation == "physical" and units == "GY" and evidence:
        label = "Physical absorbed dose (Gy)"
    elif interpretation != "unverified":
        raise ValueError("EQD2 or physical dose requires a Gy grid and documented derivation evidence.")
    ct_frame = inputs.loaded["ct"].frame_uid
    dose_frame = inputs.loaded["dose"].frame_uid
    transform_file = specification.get("ct_to_dose_transform")
    if transform_file:
        if not specification.get("spatial_evidence"):
            raise ValueError("Document dose.spatial_evidence for the supplied CT-to-dose transform.")
        transform = sitk.ReadTransform(str(transform_file))
        if transform.GetDimension() != 3:
            raise ValueError("The CT-to-dose transform must map three-dimensional LPS coordinates.")
        source = "Explicit CT LPS to dose LPS transformation."
    elif ct_frame and ct_frame == dose_frame:
        transform = sitk.Transform(3, sitk.sitkIdentity)
        source = "Identity in physical coordinates from a matching DICOM FrameOfReferenceUID."
    else:
        raise ValueError("Dose and CT frames differ or are missing. Supply a verified CT-LPS-to-dose-LPS transform.")
    provenance = {
        "units": units, "interpretation": interpretation, "label": label, "evidence": evidence,
        "alpha_beta_gy": specification.get("alpha_beta_gy"),
        "accumulation_method": specification.get("accumulation_method"),
        "treatment_timing": specification.get("treatment_timing"),
        "ct_to_dose": source, "spatial_evidence": specification.get("spatial_evidence"),
        "reported_dose_type": info.get("dose_type"),
        "dose_transform_sha256": sha256(Path(transform_file).read_bytes()).hexdigest() if transform_file else None,
    }
    return transform, provenance


def _registration_fingerprint(config, inputs, native_masks, transforms, dose_provenance):
    digest = sha256()
    for name, image in sorted((inputs.images | native_masks).items()):
        if image is not None:
            digest.update(name.encode())
            digest.update(json.dumps(geometry_summary(image), sort_keys=True).encode())
            digest.update(sitk.GetArrayViewFromImage(image).tobytes())
    for name, transform in sorted(transforms.items()):
        digest.update(name.encode())
        digest.update(transform.GetTransformEnum().to_bytes(4, byteorder="little"))
        digest.update(np.asarray(transform.GetFixedParameters(), dtype="<f8").tobytes())
        digest.update(np.asarray(transform.GetParameters(), dtype="<f8").tobytes())
        # Sample the full composition as GetParameters may expose only its back transform.
        for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (47, -36, 81)):
            digest.update(np.asarray(transform.TransformPoint(index), dtype="<f8").tobytes())
    identity = {name: {"frame": image.frame_uid, "metadata": image.metadata}
                for name, image in inputs.loaded.items()}
    review_context = {
        "thresholds": config["registration"]["thresholds"], "landmarks": config.get("landmarks"),
        "exclude_negative_rcbv": config["rcbv"]["exclude_negative"],
    }
    for item in (identity, dose_provenance, review_context):
        digest.update(json.dumps(serializable(item), sort_keys=True).encode())
    return digest.hexdigest()


def _review(config, fingerprint, transforms, unconfirmed):
    if config["mode"] == "synthetic":
        truth = config.get("synthetic_truth", {}).get("fixed_to_moving", {})
        errors = {}
        for name in ("ct", "october"):
            expected = truth.get(name)
            landmarks = _landmarks(config, name)
            if expected is None or landmarks is None:
                raise ValueError("Synthetic review requires known transforms and landmarks.")
            fixed_points = np.asarray(landmarks["fixed_lps"])
            center = np.asarray(expected["center_lps_mm"])
            matrix = np.asarray(expected["matrix"])
            translation = np.asarray(expected["translation_mm"])
            moving = (fixed_points - center) @ matrix.T + center + translation
            if not np.allclose(moving, landmarks["moving_lps"], atol=1e-7, rtol=0):
                raise ValueError(f"Synthetic landmark truth disagrees with the {name} affine.")
            estimated = np.array([transforms[name].TransformPoint(tuple(point))
                                  for point in landmarks["fixed_lps"]])
            errors[name] = float(np.max(np.linalg.norm(estimated - moving, axis=1)))
        tolerance_mm = config.get("synthetic_truth", {}).get("tre_tolerance_mm", 1.5)
        passed = all(error <= tolerance_mm for error in errors.values())
        return {"status": "synthetic_verified" if passed else "rejected",
                "max_tre_mm": errors, "tolerance_mm": tolerance_mm}
    review = config["registration"]["review"]
    if review.get("status") == "rejected":
        return {"status": "rejected", "notes": review.get("notes")}
    if unconfirmed:
        return {"status": "pending",
                "notes": f"Review same-frame rCBV alignment and confirm these sessions: {', '.join(unconfirmed)}."}
    if review.get("status") == "approved":
        if all(review.get(key) for key in ("reviewer", "notes")) and review.get("fingerprint") == fingerprint:
            return {"status": "approved", "reviewer": review["reviewer"], "notes": review["notes"]}
        return {"status": "pending", "notes": "Approval needs reviewer, notes, and the matching fingerprint."}
    return {"status": "pending", "notes": "Review registration and dose figures before approving."}


def register_inputs(config, inputs, output_dir) -> Registered:
    """Register to May anatomy, resample scalar data once, and export review evidence."""
    import matplotlib.pyplot as plt

    from .visualization import dose_overlay_figure, registration_figure, save_figure

    output = _output_directory(config, output_dir)
    original = inputs.images
    fixed = original["may_anatomy"]
    settings = config["registration"]
    transforms = {}
    aligned = {"may_anatomy": fixed}
    coverage = {}
    metrics = {}
    figures = []
    unconfirmed = []

    def save_qa(figure, name):
        figures.append(save_figure(figure, output / "qa" / f"{name}.png"))
        plt.close(figure)

    def thresholds(name):
        defaults = {key: value for key, value in settings["thresholds"].items()
                    if not isinstance(value, dict)}
        return defaults | settings["thresholds"].get(name, {})

    native_masks = {
        "fixed": _native_mask(config, "brain_mask", fixed),
        "ct": _native_mask(config, "ct_brain_mask", original["ct"]),
        "october": _native_mask(config, "october_brain_mask", original["october_anatomy"]),
    }

    for name, moving_name in (("ct", "ct"), ("october", "october_anatomy")):
        moving = original[moving_name]
        result = register_rigid(
            fixed, moving, metric=settings["metrics"].get(name, "mutual_information"),
            fixed_mask=native_masks["fixed"], moving_mask=native_masks[name],
            options=settings["options"].get(name),
        )
        transforms[name] = result.transform
        aligned[moving_name], coverage[moving_name] = resample_scalar(moving, fixed, result.transform)
        metrics[name] = {
            "optimizer": result.summary(),
            "qa": evaluate_registration(
                fixed, moving, result.transform, fixed_mask=native_masks["fixed"],
                moving_mask=native_masks[name], landmarks=_landmarks(config, name),
                thresholds=thresholds(name),
            ),
        }
        mask = native_masks[name]
        save_qa(registration_figure(
            fixed, moving, aligned[moving_name], title=f"{name.title()} anatomy to May MRI",
            before=resample_scalar(moving, fixed)[0], fixed_mask=native_masks["fixed"],
            moving_mask_registered=resample_mask(mask, fixed, result.transform) if mask is not None else None,
        ), f"{name}_registration")

    for session in ("may", "october"):
        name = f"{session}_rcbv"
        anatomy_name = f"{session}_anatomy"
        moving = original[name]
        anatomy = original[anatomy_name]
        same_frame = (inputs.loaded[name].frame_uid
                      and inputs.loaded[name].frame_uid == inputs.loaded[anatomy_name].frame_uid)
        method = settings["rcbv_method"].get(session, "identity")
        if method == "identity" and same_frame:
            local = sitk.Transform(3, sitk.sitkIdentity)
            confirmed = bool(settings["same_frame_rcbv"].get(session))
            if not confirmed:
                unconfirmed.append(session)
            details = {"method": "Identity in shared DICOM physical frame", "confirmed": confirmed}
        elif method == "rigid":
            local_result = register_rigid(
                anatomy, moving, metric=settings["metrics"].get(name, "mutual_information"),
                options=settings["options"].get(name),
            )
            local = local_result.transform
            details = local_result.summary()
        else:
            raise ValueError(
                f"{name}: the DICOM frames do not support identity; explicitly configure "
                f"registration.rcbv_method.{session}: rigid."
            )
        metrics[name] = {
            "optimizer": details, "qa": evaluate_registration(
                anatomy, moving, local, landmarks=_landmarks(config, name),
                thresholds=thresholds(name),
            ),
        }
        final = compose_transforms(transforms["october"], local) if session == "october" else local
        transforms[name] = final
        aligned[name], coverage[name] = resample_scalar(
            moving, fixed, final, allow_negative=not config["rcbv"]["exclude_negative"],
        )
        local_aligned = aligned[name] if session == "may" else resample_scalar(moving, anatomy, local)[0]
        save_qa(registration_figure(
            anatomy, moving, local_aligned, title=f"{session.title()} rCBV to session anatomy",
            before=resample_scalar(moving, anatomy)[0],
        ), f"{name}_registration")

    transforms["ct_to_dose"], dose_provenance = _dose_chain(config, inputs)
    transforms["dose"] = compose_transforms(transforms["ct"], transforms["ct_to_dose"])
    aligned["dose_on_ct"], coverage["dose_on_ct"] = resample_scalar(
        original["dose"], original["ct"], transforms["ct_to_dose"], allow_negative=False,
    )
    aligned["dose"], coverage["dose"] = resample_scalar(
        original["dose"], fixed, transforms["dose"], allow_negative=False,
    )
    for name, anatomy, dose in (
        ("ct", original["ct"], aligned["dose_on_ct"]),
        ("may", fixed, aligned["dose"]),
        ("october", aligned["october_anatomy"], aligned["dose"]),
    ):
        save_qa(dose_overlay_figure(
            anatomy, dose, dose_label=dose_provenance["label"], title=f"RTDOSE on {name.title()}",
        ), f"dose_on_{name}")

    for name, image in coverage.items():
        if not np.any(sitk.GetArrayViewFromImage(image)):
            raise ValueError(f"{name} has no valid coverage in its target geometry.")
    fingerprint = _registration_fingerprint(config, inputs, native_masks, transforms, dose_provenance)
    report = {
        "reference": "May anatomical MRI; DICOM LPS mm",
        "mapping": "Output/fixed LPS mm to input/moving LPS mm",
        "metrics": metrics, "dose_provenance": dose_provenance,
        "coverage_fraction": {name: float(np.mean(sitk.GetArrayViewFromImage(image)))
                              for name, image in coverage.items()},
        "fingerprint": fingerprint,
        "review": _review(config, fingerprint, transforms, unconfirmed),
    }
    for name, transform in transforms.items():
        folder = output / "qa"
        folder.mkdir(exist_ok=True)
        sitk.WriteTransform(transform, str(folder / f"{name}_fixed_to_moving.tfm"))
    write_json(output / "quantitative_results" / "registration.json", report)
    return Registered(aligned, coverage, transforms, report, figures)


def analyze_inputs(config, inputs, registered, output_dir) -> Analysis:
    """Compute measured perfusion changes and dose associations after spatial review."""
    _output_directory(config, output_dir)
    if registered.report["review"]["status"] not in {"approved", "synthetic_verified"}:
        raise ValueError("Registration review is required before quantitative analysis. See registration.json.")
    scaling_evidence = config["rcbv"].get("quantitative_evidence")
    normalization_settings = config["normalization"]
    if config["mode"] == "patient":
        if not scaling_evidence or not normalization_settings.get("reference_description"):
            raise ValueError("Document rCBV quantitative scaling and the reviewed normalization reference ROI.")
    paths = config["paths"]
    if not paths.get("brain_mask") or not paths.get("reference_roi"):
        raise ValueError("Supply reviewed May-space brain_mask and reference_roi paths for analysis.")
    dose_provenance = registered.report["dose_provenance"]
    if dose_provenance["units"] == "RELATIVE" and not config["dose"].get("allow_relative_analysis"):
        raise ValueError(
            "Relative dose requires explicitly configured native-unit thresholds/bins "
            "and dose.allow_relative_analysis: true."
        )
    fixed = registered.reference
    brain = load_mask(paths["brain_mask"], fixed)
    reference_roi = load_mask(paths["reference_roi"], fixed)
    exclusions = load_mask(paths["exclusions"], fixed) if paths.get("exclusions") else None
    aligned = registered.images
    may, october, dose = (sitk.GetArrayViewFromImage(aligned[name])
                           for name in ("may_rcbv", "october_rcbv", "dose"))
    valid, roi = analysis_masks(
        brain, reference_roi, may, october, exclusions=exclusions,
        may_coverage=sitk.GetArrayViewFromImage(registered.coverage["may_rcbv"]) != 0,
        october_coverage=sitk.GetArrayViewFromImage(registered.coverage["october_rcbv"]) != 0,
    )
    may_norm, october_norm, normalization = normalize(
        may, october, valid, roi, method=normalization_settings["method"],
        min_roi_voxels=normalization_settings["min_roi_voxels"],
    )
    normalization["reference_description"] = normalization_settings.get("reference_description") or (
        "Analytic reference region in the synthetic fixture."
    )
    normalization["quantitative_evidence"] = scaling_evidence or "Known synthetic perfusion scale."
    settings = config["analysis"]
    maps = change_maps(may_norm, october_norm, valid, baseline_floor=settings["baseline_floor"])
    maps.update(may_normalized=may_norm, october_normalized=october_norm)
    voxel_ml = float(np.prod(fixed.GetSpacing()) / 1000)
    options = {key: settings[key] for key in ("percent", "robust_sigma", "min_region_ml")}
    candidate = candidates(maps, valid, roi, voxel_ml=voxel_ml, absolute=settings["absolute"], **options)
    dose_coverage = sitk.GetArrayViewFromImage(registered.coverage["dose"]) != 0
    dose_valid = valid & dose_coverage & np.isfinite(dose)
    dose_report = analyze_dose(
        dose, maps, valid, candidate, voxel_ml=voxel_ml, thresholds=settings["dose_thresholds"],
        bin_edges=settings["dose_bin_edges"], dose_coverage=dose_coverage,
    )
    native_dose = sitk.GetArrayViewFromImage(inputs.images["dose"])
    dose_report["summary"]["native_grid_dose"] = summarize(native_dose)
    threshold = settings["overlay_dose_threshold"]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("The overlay dose threshold must be finite and nonnegative.")
    spatial_overlap = np.where(dose_valid & (dose >= threshold), candidate, 0).astype(np.int8)
    regions = candidate_regions(candidate, maps, voxel_ml=voxel_ml, dose=dose, dose_coverage=dose_coverage)
    for row in regions:
        row["centroid_lps_mm"] = fixed.TransformContinuousIndexToPhysicalPoint(
            tuple(reversed(row["centroid_z_y_x"])),
        )
    sensitivity = threshold_sensitivity(
        maps, valid, roi, absolute_thresholds=settings["sensitivity"], voxel_ml=voxel_ml,
        dose=dose, dose_coverage=dose_coverage, dose_threshold=threshold, **options,
    )
    change_summary = {
        **{key: summarize(maps[key][valid]) for key in ("signed", "magnitude", "percent", "log_ratio")},
        "voxel_ml": voxel_ml, "brain_mask_ml": int(brain.sum()) * voxel_ml,
        "common_brain_ml": int(valid.sum()) * voxel_ml,
        "reviewed_reference_voxels": int(reference_roi.sum()),
        "evaluable_reference_voxels": int(roi.sum()),
        "reference_coverage_fraction": float(roi.sum() / reference_roi.sum()),
        "increase_ml": int(np.count_nonzero(candidate > 0)) * voxel_ml,
        "decrease_ml": int(np.count_nonzero(candidate < 0)) * voxel_ml,
        "dose_overlap_increase_ml": int(np.count_nonzero(spatial_overlap > 0)) * voxel_ml,
        "dose_overlap_decrease_ml": int(np.count_nonzero(spatial_overlap < 0)) * voxel_ml,
        "thresholds": {"absolute": settings["absolute"], "baseline_floor": settings["baseline_floor"],
                       "overlay_dose_threshold": threshold, **options},
    }
    if paths.get("anatomical_roi"):
        anatomical_roi = load_mask(paths["anatomical_roi"], fixed)
        change_summary["anatomical_roi"] = {
            "evaluable_volume_ml": int(np.count_nonzero(anatomical_roi & valid)) * voxel_ml,
            "signed_change": summarize(maps["signed"][anatomical_roi & valid]),
            "dose": summarize(dose[anatomical_roi & dose_valid]),
        }
    limitations = [
        "Spatial associations are descriptive within one aligned volume. Nearby voxels are correlated.",
        "Perfusion direction and dose overlap can identify candidate abnormalities. "
        "Clinical interpretation requires independent evidence of injury, recurrence, or mixed pathology.",
        "Rigid transforms cannot account for anatomical deformation. Inspect local alignment near candidate regions.",
        "Linear interpolation changes local extrema and mixes neighboring samples. Coverage masks mark excluded samples.",
        "Separate normalization can suppress true global perfusion effects; shared-reference and already-normalized "
        "strategies require comparable input scaling.",
        "Reference tissue and exclusions require anatomical review, including contamination by CSF, "
        "vessels, lesions and susceptibility artifacts.",
    ]
    if dose_provenance["interpretation"] == "eqd2" and dose_provenance["reported_dose_type"] != "EFFECTIVE":
        limitations.append("The DICOM DoseType tag does not specify EFFECTIVE dose. The EQD2 label relies on "
                           "the recorded provenance evidence.")
    if dose_provenance["interpretation"] == "unverified":
        limitations.append("Biological dose derivation is unverified. Numeric values reflect DICOM scaling.")
    if not dose_provenance["treatment_timing"]:
        limitations.append("Treatment dates relative to the May and October imaging have not been verified.")
    if dose_provenance["interpretation"] in {"eqd2", "unverified"}:
        if not dose_provenance["alpha_beta_gy"] or not dose_provenance["accumulation_method"]:
            limitations.append("Alpha/beta ratio or accumulation method was not supplied.")
    if config["mode"] == "synthetic":
        limitations.append("Synthetic DICOM and analytic landmarks were used. Patient data were not evaluated.")
    report = {
        "mode": config["mode"], "inspection": inputs.report, "registration": registered.report,
        "normalization": normalization, "change": change_summary, "dose": dose_report,
        "dose_label": dose_provenance["label"], "sensitivity": sensitivity, "regions": regions,
        "limitations": limitations,
    }
    return Analysis(maps, candidate, valid, roi, dose_valid, spatial_overlap, report)


def make_figures(config, registered, analysis, output_dir) -> list[Path]:
    """Create publication PNGs on synchronized radiological coordinates."""
    import matplotlib.pyplot as plt
    from .visualization import (
        candidate_figure, dose_association_figure, image_preview, longitudinal_figure,
        normalization_figure, save_figure,
    )

    output = _output_directory(config, output_dir) / "figures"
    fixed = registered.reference
    maps = analysis.maps
    aligned = registered.images
    converted = {name: array_image(maps[name], fixed)
                 for name in ("may_normalized", "october_normalized", "signed", "percent")}
    candidate = array_image(analysis.candidate, fixed)
    roi = array_image(analysis.roi.astype(np.uint8), fixed)
    anatomical_roi = _native_mask(config, "anatomical_roi", fixed)
    label = analysis.report["dose_label"]
    threshold = config["analysis"]["overlay_dose_threshold"]
    written = []

    def save(name, figure):
        written.append(save_figure(figure, output / f"{name}.png"))
        plt.close(figure)

    save("normalization", normalization_figure(
        fixed, aligned["may_rcbv"], aligned["october_rcbv"],
        converted["may_normalized"], converted["october_normalized"], roi,
    ))
    save("longitudinal", longitudinal_figure(
        fixed, converted["may_normalized"], converted["october_normalized"],
        converted["signed"], converted["percent"], candidate, dose=aligned["dose"],
        dose_label=label, dose_levels=config["analysis"]["dose_thresholds"],
        centering_mask=array_image((analysis.candidate != 0).astype(np.uint8), fixed),
    ))
    save("candidate_overlay", candidate_figure(
        fixed, converted["signed"], candidate, dose=aligned["dose"], dose_label=label,
        dose_threshold=threshold, anatomical_roi=anatomical_roi,
    ))
    save("dose_association", dose_association_figure(
        sitk.GetArrayViewFromImage(aligned["dose"]), maps["signed"], maps["magnitude"],
        maps["percent"], analysis.dose_valid, config["analysis"]["dose_bin_edges"],
        dose_label=label,
    ))
    for direction, cmap in (("positive", "Oranges"), ("negative", "Blues")):
        save(f"{direction}_change", image_preview(
            array_image(np.abs(maps[direction]), fixed),
            title=f"{'Increased' if direction == 'positive' else 'Decreased'} rCBV",
            label="Magnitude of directional ΔnCBV", cmap=cmap,
        ))
    return written


def export_results(config, inputs, registered, analysis, output_dir, *, figures=None) -> Path:
    """Save registered NIfTI volumes, change maps, figures, and quantitative reports."""
    output = _output_directory(config, output_dir)
    image_dir = output / "registered_images"
    image_dir.mkdir(exist_ok=True)
    for name, image in registered.images.items():
        sitk.WriteImage(image, str(image_dir / f"{name}.nii.gz"))
    for name, image in registered.coverage.items():
        sitk.WriteImage(image, str(image_dir / f"{name}_valid_coverage.nii.gz"))
    arrays = analysis.maps | {
        "candidate": analysis.candidate, "candidate_dose_overlap": analysis.spatial_overlap,
        "analysis_valid": analysis.valid.astype(np.uint8), "reference_roi": analysis.roi.astype(np.uint8),
        "dose_analysis_valid": analysis.dose_valid.astype(np.uint8),
    }
    for name, array in arrays.items():
        sitk.WriteImage(array_image(array, registered.reference), str(image_dir / f"{name}.nii.gz"))
    quantitative = output / "quantitative_results"
    report = analysis.report
    write_json(quantitative / "analysis.json", report)
    for filename, rows in {
        "dose_thresholds": report["dose"]["thresholds"],
        "dose_bins": report["dose"]["bins"],
        "dose_associations": [{"map": name, **row} for name, row in report["dose"]["associations"].items()],
        "threshold_sensitivity": report["sensitivity"],
        "candidate_regions": report["regions"],
    }.items():
        write_csv(quantitative / f"{filename}.csv", rows)
    if figures is None:
        figures = make_figures(config, registered, analysis, output)
    return write_report(output / "reports" / "summary.md", report, [*registered.figures, *figures])
