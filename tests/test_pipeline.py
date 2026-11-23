from types import SimpleNamespace

import pytest
import SimpleITK as sitk

from necrosis_analysis.pipeline import (
    _dose_chain, _review, analyze_inputs, load_config,
)


def _dose_inputs(ct_frame, dose_frame, *, dose_type="PHYSICAL", frames=4):
    image = sitk.Image([5, 5, frames], sitk.sitkFloat32)
    return SimpleNamespace(
        loaded={
            "ct": SimpleNamespace(frame_uid=ct_frame),
            "dose": SimpleNamespace(
                frame_uid=dose_frame, metadata={"dose_units": "GY", "dose_type": dose_type},
            ),
        },
        images={"dose": image},
    )


def test_patient_review_is_tied_to_artifacts_and_alignment(tmp_path):
    config = load_config(overrides={
        "registration": {"review": {
            "status": "approved", "reviewer": "Research reviewer",
            "notes": "Reviewed all three planes", "fingerprint": "reviewed",
        }},
    })
    assert _review(config, "new result", {}, [])["status"] == "pending"
    assert _review(config, "reviewed", {}, ["may"])["status"] == "pending"
    assert _review(config, "reviewed", {}, [])["status"] == "approved"
    rejected = SimpleNamespace(report={"review": {"status": "rejected"}})
    with pytest.raises(ValueError, match="Registration review is required"):
        analyze_inputs(config, None, rejected, tmp_path)


def test_dose_provenance_and_coordinate_gate(tmp_path):
    config = load_config()
    transform, provenance = _dose_chain(config, _dose_inputs("1.2", "1.2"))
    assert transform.TransformPoint((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert "derivation unverified" in provenance["label"]
    with pytest.raises(ValueError, match="frames differ"):
        _dose_chain(config, _dose_inputs("1.2", "1.3"))
    with pytest.raises(ValueError, match="DoseType ERROR"):
        _dose_chain(config, _dose_inputs("1.2", "1.2", dose_type="ERROR"))
    with pytest.raises(ValueError, match="multiple distinct RTDOSE frames"):
        _dose_chain(config, _dose_inputs("1.2", "1.2", frames=1))

    supplied = tmp_path / "ct_to_dose.tfm"
    sitk.WriteTransform(sitk.TranslationTransform(3, (2, -3, 5)), str(supplied))
    config["dose"].update(ct_to_dose_transform=str(supplied), spatial_evidence="Known fixture transform")
    transform, _ = _dose_chain(config, _dose_inputs("1.2", "1.3"))
    assert transform.TransformPoint((1.0, 2.0, 3.0)) == (3.0, -1.0, 8.0)

    config["dose"]["interpretation"] = "eqd2"
    with pytest.raises(ValueError, match="documented derivation evidence"):
        _dose_chain(config, _dose_inputs("1.2", "1.3"))
    config["dose"]["evidence"] = "Known fixture dose derivation"
    _, provenance = _dose_chain(config, _dose_inputs("1.2", "1.3", dose_type="EFFECTIVE"))
    assert provenance["label"] == "Accumulated EQD2 (Gy)"
