import matplotlib
import numpy as np
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from necrosis_analysis.visualization import (
    PLANES,
    _point,
    _slice,
    candidate_figure,
    make_viewer,
    registration_figure,
    save_figure,
)


def _oblique_images():
    z, y, x = np.indices((13, 17, 19))
    theta = np.deg2rad(21)
    direction = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    spacing = np.array([1.2, 1.6, 2.5])
    origin = np.array([-36, 59, 112])
    lps = np.einsum("ij,jzyx->izyx", direction @ np.diag(spacing), [x, y, z]) + origin[:, None, None, None]

    image = sitk.GetImageFromArray((0.2 * lps[0] + 0.4 * lps[1] + 0.9 * lps[2]).astype(np.float32))
    image.SetDirection(direction.ravel().tolist())
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(origin.tolist())
    labels = np.zeros(x.shape, dtype=np.int8)
    labels[5:9, 6:12, 6:10] = -1
    mask = sitk.GetImageFromArray(labels)
    mask.CopyInformation(image)
    return image, mask


@pytest.mark.parametrize("name", list(PLANES))
def test_display_planes_sample_correct_physical_points(name):
    image, _ = _oblique_images()
    plane = PLANES[name]
    display = _slice(image, plane, _point(image)[plane.normal])
    pixel = len(display.x) // 2, len(display.y) // 2
    point = display.point(*pixel)
    assert point[plane.normal] == display.location
    expected = 0.2 * point[0] + 0.4 * point[1] + 0.9 * point[2]
    assert np.isclose(display.sample(image)[pixel[1], pixel[0]], expected, atol=2e-5)
    assert display.x[-1] > display.x[0]
    assert (display.y[-1] > display.y[0]) == (plane.vertical_step > 0)


def test_registration_candidate_and_viewer_figures(tmp_path):
    anatomy, candidate = _oblique_images()
    signed = sitk.Cast(candidate, sitk.sitkFloat32)
    registration = registration_figure(anatomy, anatomy, anatomy, title="Synthetic registration")
    try:
        assert save_figure(registration, tmp_path / "qa" / "registration.png").stat().st_size > 1000
    finally:
        plt.close(registration)

    figure = candidate_figure(anatomy, signed, candidate)
    try:
        overlay = figure.axes[0].images[-1].get_array()
        assert np.any(overlay.compressed() == 1), "Negative-only candidate labels should be visible."
    finally:
        plt.close(figure)

    static = make_viewer(anatomy, anatomy, anatomy, signed, interactive=False)
    try:
        assert len(static.axes) == 4
        assert all(len(axis.images) == 2 for axis in static.axes[:3])
    finally:
        plt.close(static)


def test_viewer_updates_planes_and_synchronized_plot():
    anatomy, labels = _oblique_images()
    signed = sitk.Cast(labels, sitk.sitkFloat32)
    viewer = make_viewer(anatomy, anatomy, anatomy, signed)
    plane, position, synchronized = viewer.children[0].children
    probe, image = viewer.children[-2:]
    first_plot = image.value.tobytes()
    assert first_plot.startswith(b"\x89PNG")
    assert "reference voxel" in probe.value

    plane.value = "Sagittal"
    assert position.description == "Slice x (mm)"
    assert image.value.tobytes() != first_plot

    one_panel = image.value.tobytes()
    synchronized.value = True
    assert image.value.tobytes() != one_panel
