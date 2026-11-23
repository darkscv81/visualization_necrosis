"""Physical volume and descriptive dose–perfusion association measurements."""

import numpy as np
from scipy.stats import rankdata

from .normalization import summarize


def _association(dose: np.ndarray, perfusion: np.ndarray) -> dict:
    valid = np.isfinite(perfusion)
    dose, perfusion = dose[valid], perfusion[valid]
    result = {"n": int(dose.size), "pearson_r": None, "spearman_rho": None}
    if dose.size >= 2 and np.ptp(dose) > 0 and np.ptp(perfusion) > 0:
        result["pearson_r"] = float(np.corrcoef(dose, perfusion)[0, 1])
        result["spearman_rho"] = float(np.corrcoef(rankdata(dose), rankdata(perfusion))[0, 1])
    return result


def analyze_dose(
    dose: np.ndarray,
    maps: dict[str, np.ndarray],
    valid: np.ndarray,
    candidate: np.ndarray,
    *,
    voxel_ml: float,
    thresholds: list[float],
    bin_edges: list[float],
    dose_coverage: np.ndarray | None = None,
) -> dict:
    """Evaluate the dose on common brain/rCBV support within the dose-grid coverage.

    Native numeric dose units are carried through unchanged. The caller verifies
    scaling, units and accumulation provenance. Associations are descriptive.
    """
    map_names = ("signed", "magnitude", "percent")
    if dose.ndim != 3 or any(a.shape != dose.shape for a in (valid, candidate, *(maps[n] for n in map_names))):
        raise ValueError("Dose analysis requires 3D dose, change maps, and masks on one grid")
    if valid.dtype != bool or not np.isin(candidate, [-1, 0, 1]).all() or np.any((candidate != 0) & ~valid):
        raise ValueError("Candidates must use {-1, 0, 1} labels inside valid common rCBV support")
    if not np.isfinite(voxel_ml) or voxel_ml <= 0:
        raise ValueError("voxel_ml must be finite and strictly positive")
    if any(not np.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("Dose thresholds must be finite and nonnegative")
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Dose bin edges must contain at least two finite, strictly increasing values")

    evaluable = valid & np.isfinite(dose)
    if dose_coverage is not None:
        if dose_coverage.shape != dose.shape or dose_coverage.dtype != bool:
            raise ValueError("Dose coverage must be a boolean mask on the common grid")
        evaluable &= dose_coverage
    sampled_dose = dose[evaluable]
    if not sampled_dose.size:
        raise ValueError("No dose samples overlap the valid common brain and rCBV coverage")
    if (sampled_dose < 0).any():
        raise ValueError("The evaluable dose contains negative values. Review the dose source and scaling")

    selected = candidate != 0
    n_candidate = int(np.count_nonzero(selected))
    sampled_candidate = candidate[evaluable]
    sampled_selected = sampled_candidate != 0
    n_evaluable_candidate = int(np.count_nonzero(sampled_selected))
    dose_stats = summarize(sampled_dose)
    candidate_stats = summarize(sampled_dose[sampled_selected])
    summary = {
        "dose": dose_stats,
        "candidate_dose": candidate_stats,
        "increased_candidate_dose": summarize(sampled_dose[sampled_candidate > 0]),
        "decreased_candidate_dose": summarize(sampled_dose[sampled_candidate < 0]),
        "valid_brain_ml": int(np.count_nonzero(valid)) * voxel_ml,
        "dose_evaluable_brain_ml": int(sampled_dose.size) * voxel_ml,
        "candidate_ml": n_candidate * voxel_ml,
        "dose_evaluable_candidate_ml": n_evaluable_candidate * voxel_ml,
        "candidate_without_dose_ml": (n_candidate - n_evaluable_candidate) * voxel_ml,
    }
    threshold_rows = []
    for threshold in thresholds:
        high = sampled_dose >= threshold
        overlap = high & sampled_selected
        n_overlap = int(np.count_nonzero(overlap))
        high_dose_stats = summarize(sampled_dose[high])
        overlap_dose_stats = summarize(sampled_dose[overlap])
        threshold_rows.append({
            "threshold": float(threshold),
            "within_observed_dose_range": bool(dose_stats["min"] <= threshold <= dose_stats["max"]),
            "volume_at_least_threshold_ml": int(np.count_nonzero(high)) * voxel_ml,
            "candidate_ml": n_candidate * voxel_ml,
            "dose_evaluable_candidate_ml": n_evaluable_candidate * voxel_ml,
            "overlap_ml": n_overlap * voxel_ml,
            "candidate_pct_minimum": 100 * n_overlap / n_candidate if n_candidate else None,
            "dose_evaluable_candidate_pct": 100 * n_overlap / n_evaluable_candidate if n_evaluable_candidate else None,
            "mean_candidate_dose": candidate_stats["mean"],
            "median_candidate_dose": candidate_stats["median"],
            "mean_dose_at_least_threshold": high_dose_stats["mean"],
            "median_dose_at_least_threshold": high_dose_stats["median"],
            "mean_overlap_dose": overlap_dose_stats["mean"],
            "median_overlap_dose": overlap_dose_stats["median"],
        })

    sampled_maps = {name: maps[name][evaluable] for name in map_names}
    indices = np.searchsorted(edges, sampled_dose, side="right") - 1
    indices[sampled_dose == edges[-1]] = len(edges) - 2
    bins = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = indices == index
        signed_stats = summarize(sampled_maps["signed"][in_bin])
        magnitude_stats = summarize(sampled_maps["magnitude"][in_bin])
        percent_stats = summarize(sampled_maps["percent"][in_bin])
        bins.append({
            "lower": float(lower),
            "upper": float(upper),
            "upper_inclusive": index == len(edges) - 2,
            "voxels": int(np.count_nonzero(in_bin)),
            "mean_signed_change": signed_stats["mean"],
            "median_signed_change": signed_stats["median"],
            "signed_change_p25": signed_stats["p25"],
            "signed_change_p75": signed_stats["p75"],
            "mean_absolute_change": magnitude_stats["mean"],
            "median_absolute_change": magnitude_stats["median"],
            "percent_change_voxels": percent_stats["n"],
            "mean_percent_change": percent_stats["mean"],
            "median_percent_change": percent_stats["median"],
        })
    summary["voxels_outside_dose_bins"] = int(np.count_nonzero((indices < 0) | (indices >= len(bins))))
    return {
        "summary": summary,
        "thresholds": threshold_rows,
        "bins": bins,
        "associations": {name: _association(sampled_dose, values) for name, values in sampled_maps.items()},
    }
