import numpy as np
import pytest

from necrosis_analysis.dose_analysis import analyze_dose


def test_overlap_and_correlation_on_partially_covered_grid():
    dose = np.array([[[0.0, 10.0, 20.0, 30.0, 400.0, 500.0, 600.0, 800.0]]])
    signed = np.array([[[-2.0, -1.0, 0.0, 1.0, 50.0, 70.0, 90.0, 95.0]]])
    maps = {"signed": signed, "magnitude": np.abs(signed), "percent": 25 * signed}
    maps["percent"][0, 0, 3] = np.nan
    valid = np.array([[[True, True, True, True, True, False, False, False]]])
    coverage = np.array([[[True, True, True, True, False, True, True, True]]])
    candidate = np.array([[[-1, -1, 0, 1, 1, 0, 0, 0]]])
    output = analyze_dose(
        dose, maps, valid, candidate, voxel_ml=0.012,
        thresholds=[0.0, 20.0, 800.0], bin_edges=[0.0, 10.0, 30.0], dose_coverage=coverage,
    )
    assert output["summary"]["dose"]["n"] == 4
    assert output["summary"]["dose"]["max"] == 30.0
    assert output["summary"]["candidate_dose"]["median"] == 10.0
    assert output["summary"]["candidate_without_dose_ml"] == pytest.approx(0.012, abs=1e-12)
    threshold = output["thresholds"][1]
    assert threshold["volume_at_least_threshold_ml"] == pytest.approx(0.024, abs=1e-12)
    assert threshold["overlap_ml"] == pytest.approx(0.012, abs=1e-12)
    assert threshold["candidate_pct_minimum"] == 25.0
    assert threshold["dose_evaluable_candidate_pct"] == pytest.approx(100 / 3, abs=1e-12)
    assert not output["thresholds"][2]["within_observed_dose_range"]
    assert output["thresholds"][2]["mean_dose_at_least_threshold"] is None
    assert [row["voxels"] for row in output["bins"]] == [1, 3]
    assert output["bins"][1]["percent_change_voxels"] == 2
    assert output["summary"]["voxels_outside_dose_bins"] == 0
    assert output["associations"]["signed"]["pearson_r"] == pytest.approx(1.0, abs=1e-12)
    assert output["associations"]["signed"]["spearman_rho"] == pytest.approx(1.0, abs=1e-12)
    assert output["associations"]["percent"]["n"] == 3


def test_constant_dose_and_empty_candidate_have_undefined_statistics():
    dose = np.zeros((1, 1, 4))
    maps = {key: np.arange(4).reshape(dose.shape) for key in ("signed", "magnitude", "percent")}
    output = analyze_dose(
        dose, maps, np.ones(dose.shape, dtype=bool), np.zeros(dose.shape, dtype=np.int8),
        voxel_ml=1.0, thresholds=[0.0], bin_edges=[0.0, 1.0],
    )
    assert output["associations"]["signed"]["pearson_r"] is None
    assert output["associations"]["signed"]["spearman_rho"] is None
    assert output["thresholds"][0]["dose_evaluable_candidate_pct"] is None
    assert output["summary"]["candidate_dose"]["median"] is None


def test_no_dose_overlap_fails_explicitly():
    dose = np.full((1, 1, 3), np.nan)
    maps = {name: dose.copy() for name in ("signed", "magnitude", "percent")}
    with pytest.raises(ValueError, match="No dose samples overlap"):
        analyze_dose(
            dose, maps, np.ones(dose.shape, dtype=bool), np.zeros(dose.shape, dtype=np.int8),
            voxel_ml=1, thresholds=[0.0], bin_edges=[0.0, 1.0],
        )
