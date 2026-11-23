"""Physical LPS image planes, registration QA, and longitudinal figures."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_gradient_magnitude


@dataclass(frozen=True)
class Plane:
    name: str
    horizontal: int
    vertical: int
    normal: int
    vertical_step: int
    horizontal_label: str
    vertical_label: str


PLANES = {
    "Axial": Plane("Axial", 0, 1, 2, -1, "x · Right → Left", "y · Posterior → Anterior"),
    "Coronal": Plane("Coronal", 0, 2, 1, 1, "x · Right → Left", "z · Inferior → Superior"),
    "Sagittal": Plane("Sagittal", 1, 2, 0, 1, "y · Anterior → Posterior", "z · Inferior → Superior"),
}

DECREASE = "#2675b5"
INCREASE = "#dc5346"
HIGH_DOSE = "#ffbc42"
OVERLAP = "#a95bd0"


@dataclass(frozen=True)
class Slice:
    plane: Plane
    x: np.ndarray
    y: np.ndarray
    location: float
    step: tuple[float, float]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        half_x, half_y = np.array(self.step) / 2
        half_y *= self.plane.vertical_step
        return self.x[0] - half_x, self.x[-1] + half_x, self.y[0] - half_y, self.y[-1] + half_y

    def point(self, column: int, row: int) -> tuple[float, float, float]:
        point = [0.0] * 3
        point[self.plane.horizontal] = float(self.x[column])
        point[self.plane.vertical] = float(self.y[row])
        point[self.plane.normal] = float(self.location)
        return tuple(point)

    def sample(self, image: sitk.Image, *, categorical: bool = False) -> np.ndarray:
        direction = np.zeros((3, 3))
        direction[self.plane.horizontal, 0] = 1
        direction[self.plane.vertical, 1] = self.plane.vertical_step
        direction[self.plane.normal, 2] = 1
        interpolator = sitk.sitkNearestNeighbor if categorical else sitk.sitkLinear
        result = sitk.Resample(
            image,
            [len(self.x), len(self.y), 1],
            sitk.Transform(3, sitk.sitkIdentity),
            interpolator,
            self.point(0, 0),
            [*self.step, 1.0],
            direction.ravel().tolist(),
            0.0 if categorical else float("nan"),
            sitk.sitkFloat32,
        )
        return sitk.GetArrayFromImage(result)[0]


def _bounds(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Visualization requires a scalar 3-D image.")
    corners = np.asarray([
        image.TransformContinuousIndexToPhysicalPoint(index)
        for index in product(*[(-0.5, size - 0.5) for size in image.GetSize()])
    ])
    return corners.min(axis=0), corners.max(axis=0)


def _point(image: sitk.Image, mask: sitk.Image | None = None) -> tuple[float, float, float]:
    if mask is not None:
        data = sitk.GetArrayViewFromImage(mask)
        stride = max(1, int(np.ceil(np.cbrt(data.size / 250_000))))
        samples = data[::stride, ::stride, ::stride]
        points = np.argwhere(np.isfinite(samples) & (samples != 0)) * stride
        if not len(points) and stride > 1:
            points = np.argwhere(np.isfinite(data) & (data != 0))
        if len(points):
            center = points.mean(axis=0)
            index = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
            return mask.TransformIndexToPhysicalPoint(tuple(map(int, index[::-1])))
    return image.TransformContinuousIndexToPhysicalPoint(tuple((size - 1) / 2 for size in image.GetSize()))


def _slice(image: sitk.Image, plane: Plane, position: float) -> Slice:
    lower, upper = _bounds(image)
    display_step = min(image.GetSpacing())

    def centers(axis: int) -> tuple[np.ndarray, float]:
        span = upper[axis] - lower[axis]
        count = min(420, max(2, int(np.ceil(span / display_step))))
        step = span / count
        return lower[axis] + (np.arange(count) + 0.5) * step, step

    x, step_x = centers(plane.horizontal)
    y, step_y = centers(plane.vertical)
    return Slice(plane, x, y[::plane.vertical_step], position, (step_x, step_y))


def _slices(image: sitk.Image, center: tuple[float, float, float]) -> Iterable[Slice]:
    for plane in PLANES.values():
        yield _slice(image, plane, center[plane.normal])


def _location(display: Slice) -> str:
    return f"{display.plane.name} · {'xyz'[display.plane.normal]} = {display.location:.1f} mm"


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _limits(*values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    samples = []
    for array in values:
        flattened = array.reshape(-1)
        stride = max(1, int(np.ceil(flattened.size / 250_000)))
        sample = _finite(flattened[::stride])
        if not sample.size and stride > 1:
            sample = _finite(flattened)
        if sample.size:
            samples.append(sample)
    merged = np.concatenate(samples) if samples else np.array([0, 1])
    low, high = np.percentile(merged, [1, 99])
    if symmetric:
        limit = max(abs(low), abs(high))
        if limit == 0:
            limit = max(float(np.max(np.abs(merged))), 1e-6)
        return -float(limit), float(limit)
    if low == high:
        padding = max(abs(low) * 0.05, 1e-6)
        return float(low - padding), float(high + padding)
    return float(low), float(high)


def _data(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayViewFromImage(image)


def _draw(
    axis: Axes,
    image_slice: Slice,
    array: np.ndarray,
    title: str,
    *,
    cmap: str = "gray",
    limits: tuple[float, float] | None = None,
):
    axis.set_facecolor("#ededed")
    axis.set_title(title)
    axis.set_xlabel(f"{image_slice.plane.horizontal_label} (mm)")
    axis.set_ylabel(f"{image_slice.plane.vertical_label} (mm)")
    axis.tick_params(labelsize=7)
    axis.xaxis.get_major_locator().set_params(nbins=4)
    axis.yaxis.get_major_locator().set_params(nbins=4)
    return _layer(axis, image_slice, array, cmap, limits or _limits(array))


def _layer(axis: Axes, image_slice: Slice, values: np.ndarray, cmap, limits: tuple[float, float], alpha: float = 1.0):
    return axis.imshow(
        np.ma.masked_invalid(values), extent=image_slice.extent, origin="lower", aspect="equal",
        interpolation="nearest", cmap=cmap, vmin=limits[0], vmax=limits[1], alpha=alpha,
    )


def _outline(axis: Axes, image_slice: Slice, mask: np.ndarray, color: str, *, width: float = 1.0) -> None:
    binary = np.asarray(mask, dtype=bool)
    if np.any(binary) and not np.all(binary):
        axis.contour(image_slice.x, image_slice.y, binary.astype(float), levels=[0.5], colors=[color], linewidths=width)


def _dose_contours(axis: Axes, image_slice: Slice, dose: np.ndarray, levels: Iterable[float]) -> None:
    finite = _finite(dose)
    if finite.size:
        visible = sorted(set(level for level in levels if finite.min() < level < finite.max()))
        if visible:
            contours = axis.contour(
                image_slice.x, image_slice.y, np.ma.masked_invalid(dose), levels=visible,
                cmap="copper", linewidths=1.3,
            )
            axis.clabel(contours, inline=True, fontsize=8, fmt="%g")


def _colorbar(figure: Figure, plot, axes, label: str) -> None:
    figure.colorbar(plot, ax=axes, shrink=0.72, pad=0.025, label=label, extend="both")


def save_figure(figure: Figure, path: str | Path) -> Path:
    """Save a PNG or PDF figure, creating the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, facecolor="white", bbox_inches="tight")
    return path


def image_preview(image: sitk.Image, *, title: str, label: str | None = None, cmap: str = "gray") -> Figure:
    """Show three orthogonal LPS planes through the native image center."""
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4), layout="constrained")
    limits = _limits(_data(image))
    for axis, image_slice in zip(axes, _slices(image, _point(image))):
        preview = _draw(axis, image_slice, image_slice.sample(image), _location(image_slice), cmap=cmap, limits=limits)
    if label:
        _colorbar(figure, preview, axes, label)
    figure.suptitle(title)
    return figure


def _scaled(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return np.clip((values - limits[0]) / (limits[1] - limits[0]), 0.0, 1.0)


def _edges(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 5:
        return np.zeros_like(values, dtype=bool)
    filled = np.where(finite, values, np.median(values[finite]))
    gradient = gaussian_gradient_magnitude(filled, sigma=1.1)
    return finite & (gradient > np.percentile(gradient[finite], 94))


def registration_figure(
    fixed: sitk.Image,
    moving_original: sitk.Image,
    registered: sitk.Image,
    *,
    title: str,
    before: sitk.Image | None = None,
    fixed_mask: sitk.Image | None = None,
    moving_mask_registered: sitk.Image | None = None,
) -> Figure:
    """Compare identity and registered fusion at matched physical positions."""
    figure, axes = plt.subplots(3, 6, figsize=(19, 10), layout="constrained")
    before = moving_original if before is None else before
    fixed_limits = _limits(_data(fixed))
    moving_limits = _limits(_data(moving_original))
    independent_masks = fixed_mask is not None and moving_mask_registered is not None
    for row, native, display in zip(
        axes, _slices(moving_original, _point(moving_original)), _slices(fixed, _point(fixed, fixed_mask)),
    ):
        fixed_values = display.sample(fixed)
        before_values = display.sample(before)
        registered_values = display.sample(registered)
        _draw(row[0], display, fixed_values, f"Fixed · {_location(display)}", limits=fixed_limits)
        _draw(row[1], native, native.sample(moving_original), f"Moving native · {_location(native)}", limits=moving_limits)
        for axis, values, heading in zip(row[2:4], (before_values, registered_values), ("Identity fusion", "Rigid fusion")):
            _draw(axis, display, fixed_values, heading, limits=fixed_limits)
            _layer(axis, display, values, "cividis", moving_limits, alpha=0.5)

        y, x = np.indices(fixed_values.shape)
        tile_length_mm = 12
        check = ((x * display.step[0] // tile_length_mm + y * display.step[1] // tile_length_mm) % 2) == 0
        fixed_scaled = _scaled(fixed_values, fixed_limits)
        registered_scaled = _scaled(registered_values, moving_limits)
        board = np.where(check, fixed_scaled, registered_scaled)
        _draw(row[4], display, board, "Rigid checkerboard", limits=(0, 1))

        heading = "Independent mask contours" if independent_masks else "Anatomical edge contours"
        _draw(row[5], display, fixed_values, heading, limits=fixed_limits)
        fixed_edges = display.sample(fixed_mask, categorical=True) > 0 if fixed_mask is not None else _edges(fixed_values)
        moving_edges = (display.sample(moving_mask_registered, categorical=True) > 0
                        if moving_mask_registered is not None else _edges(registered_values))
        _outline(row[5], display, fixed_edges, "#66e3f0", width=0.8)
        _outline(row[5], display, moving_edges, "#ff8f65", width=0.75)
    figure.legend(handles=[Patch(color="#66e3f0", label="Fixed boundary"), Patch(color="#ff8f65", label="Moving boundary")],
                  loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle(title)
    return figure


def _require_dose_label(dose: sitk.Image | None, dose_label: str | None) -> None:
    if dose is not None and not dose_label:
        raise ValueError("Supply a dose label reflecting the verified interpretation and units.")


def dose_overlay_figure(
    anatomy: sitk.Image, dose: sitk.Image, *, dose_label: str, title: str,
) -> Figure:
    """Show the validated dose-to-anatomy spatial relationship."""
    _require_dose_label(dose, dose_label)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), layout="constrained")
    background_limits = _limits(_data(anatomy))
    finite_dose = _finite(_data(dose))
    vmax = max(float(finite_dose.max()), 1e-6) if finite_dose.size else 1.0
    for axis, display in zip(axes, _slices(anatomy, _point(anatomy))):
        _draw(axis, display, display.sample(anatomy), _location(display), limits=background_limits)
        values = display.sample(dose)
        overlay = _layer(axis, display, np.where(values > 0, values, np.nan), "magma", (0, vmax), alpha=0.67)
    _colorbar(figure, overlay, axes, dose_label)
    figure.suptitle(title)
    return figure


def _distribution(axis: Axes, may: np.ndarray, october: np.ndarray, title: str, units: str) -> None:
    may = _finite(may)
    october = _finite(october)
    if may.size and october.size:
        lower, upper = _limits(may, october)
        if (may.min() == may.max()) and (october.min() == october.max()) and may[0] == october[0]:
            padding = max(0.05 * abs(may[0]), 0.01)
            lower, upper = may[0] - padding, may[0] + padding
        edges = np.linspace(lower, upper, 38)
        for values, label, color in ((may, "May", "#2b63a0"), (october, "October", "#bd5936")):
            axis.hist(values, bins=edges, histtype="step", linewidth=1.6, label=label, color=color)
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No paired finite values", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel(units)
    axis.set_ylabel("Voxel count")


def normalization_figure(
    anatomy: sitk.Image,
    may_raw: sitk.Image,
    october_raw: sitk.Image,
    may_normalized: sitk.Image,
    october_normalized: sitk.Image,
    reference_roi: sitk.Image,
) -> Figure:
    """Show reference ROI location and paired raw and normalized distributions."""
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.4), layout="constrained")
    limits = _limits(_data(anatomy))
    for axis, display in zip(axes[0], _slices(anatomy, _point(anatomy, reference_roi))):
        _draw(axis, display, display.sample(anatomy), _location(display), limits=limits)
        mask = display.sample(reference_roi, categorical=True) > 0
        _outline(axis, display, mask, "#3bd5cd", width=1.6)
    raw_may, raw_october = _data(may_raw), _data(october_raw)
    n_may, n_october = _data(may_normalized), _data(october_normalized)
    valid = np.isfinite(n_may) & np.isfinite(n_october)
    roi = (_data(reference_roi) > 0) & valid
    axes[0, 3].axis("off")
    if np.any(roi):
        axes[0, 3].text(
            0.0, 0.5,
            f"Reference medians\n\nMay    {np.median(raw_may[roi]):.3g} raw  →  {np.median(n_may[roi]):.3g}\n"
            f"October    {np.median(raw_october[roi]):.3g} raw  →  {np.median(n_october[roi]):.3g}",
            transform=axes[0, 3].transAxes, va="center",
        )
    _distribution(axes[1, 0], raw_may[valid], raw_october[valid], "Paired coverage · Raw", "Scaled input rCBV")
    _distribution(axes[1, 1], raw_may[roi], raw_october[roi], "Reference ROI · Raw", "Scaled input rCBV")
    _distribution(axes[1, 2], n_may[valid], n_october[valid], "Paired coverage · Normalized", "Normalized rCBV")
    _distribution(axes[1, 3], n_may[roi], n_october[roi], "Reference ROI · Normalized", "Normalized rCBV")
    figure.suptitle("Reference tissue and rCBV normalization")
    return figure


def _candidate_layers(
    axis: Axes,
    display: Slice,
    signed: np.ndarray,
    candidate: np.ndarray,
    dose_mask: np.ndarray | None = None,
    *,
    alpha: float = 0.72,
) -> None:
    classes = np.full(signed.shape, np.nan)
    classes[candidate & (signed < 0)] = 1
    classes[candidate & (signed > 0)] = 2
    if dose_mask is not None:
        classes[dose_mask & ~candidate] = 3
        classes[dose_mask & candidate] = 4
        _outline(axis, display, dose_mask, HIGH_DOSE, width=0.75)
    _layer(axis, display, classes, ListedColormap([DECREASE, INCREASE, HIGH_DOSE, OVERLAP]), (0.5, 4.5), alpha=alpha)


def longitudinal_figure(
    anatomy: sitk.Image,
    may: sitk.Image,
    october: sitk.Image,
    signed: sitk.Image,
    percent: sitk.Image,
    candidate: sitk.Image,
    *,
    dose: sitk.Image | None = None,
    dose_label: str | None = None,
    dose_levels: Iterable[float] = (),
    centering_mask: sitk.Image | None = None,
) -> Figure:
    """Display matched radiological planes with common rCBV and zero-centered change scales."""
    _require_dose_label(dose, dose_label)
    dose_levels = tuple(dose_levels)
    figure, axes = plt.subplots(3, 6, figsize=(20, 10.5), layout="constrained")
    n_limits = _limits(_data(may), _data(october))
    change_limits = _limits(_data(signed), symmetric=True)
    pct_limits = _limits(_data(percent), symmetric=True)
    anatomy_limits = _limits(_data(anatomy))
    center = _point(anatomy, candidate if centering_mask is None else centering_mask)
    for row, display in zip(axes, _slices(anatomy, center)):
        may_slice = display.sample(may)
        october_slice = display.sample(october)
        delta = display.sample(signed)
        rc = _draw(row[0], display, may_slice, f"May · {_location(display)}", cmap="viridis", limits=n_limits)
        _draw(row[1], display, october_slice, "October", cmap="viridis", limits=n_limits)
        difference = _draw(row[2], display, delta, "Signed change", cmap="RdBu_r", limits=change_limits)
        relative = _draw(row[3], display, display.sample(percent), "Relative change", cmap="RdBu_r", limits=pct_limits)

        if dose is None:
            magnitude = _draw(row[4], display, np.abs(delta), "Change magnitude", cmap="magma", limits=(0, change_limits[1]))
        else:
            dose_slice = display.sample(dose)
            _draw(row[4], display, may_slice, "May with dose contours", cmap="viridis", limits=n_limits)
            _dose_contours(row[4], display, dose_slice, dose_levels)
            _dose_contours(row[2], display, dose_slice, dose_levels)
        _draw(row[5], display, display.sample(anatomy), "Candidate regions", limits=anatomy_limits)
        labels = display.sample(candidate, categorical=True)
        _candidate_layers(row[5], display, display.sample(signed, categorical=True), np.isfinite(labels) & (labels != 0))

    _colorbar(figure, rc, axes[:, :2], "Normalized rCBV")
    _colorbar(figure, difference, axes[:, 2], "Δ normalized rCBV")
    _colorbar(figure, relative, axes[:, 3], "Relative change (%)")
    if dose is None:
        _colorbar(figure, magnitude, axes[:, 4], "|Δ normalized rCBV|")
    figure.legend(handles=[Patch(color=DECREASE, label="Decreased rCBV"), Patch(color=INCREASE, label="Increased rCBV")],
                  loc="outside lower center", ncol=2, frameon=False)
    dose_caption = f" · Contours in {dose_label}" if dose is not None and dose_levels else ""
    figure.suptitle(f"Longitudinal rCBV change{dose_caption}")
    return figure


def change_direction_figure(
    anatomy: sitk.Image,
    positive: sitk.Image,
    negative: sitk.Image,
    *,
    centering_mask: sitk.Image | None = None,
) -> Figure:
    """Show separate increase and decrease maps on a matched absolute scale."""
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8), layout="constrained")
    limit = _limits(_data(positive), _data(negative), symmetric=True)[1]
    for column, display in zip(axes.T, _slices(anatomy, _point(anatomy, centering_mask))):
        increased = display.sample(positive)
        decreased = display.sample(negative)
        up = _draw(column[0], display, np.where(increased > 0, increased, np.nan),
                   _location(display), cmap="Reds", limits=(0, limit))
        down = _draw(column[1], display, np.where(decreased < 0, -decreased, np.nan),
                     _location(display), cmap="Blues", limits=(0, limit))
    _colorbar(figure, up, axes[0], "Increase (Δ normalized rCBV)")
    _colorbar(figure, down, axes[1], "Decrease magnitude (|Δ normalized rCBV|)")
    figure.suptitle("Direction of longitudinal rCBV change")
    return figure


def candidate_figure(
    anatomy: sitk.Image,
    signed: sitk.Image,
    candidate: sitk.Image,
    *,
    dose: sitk.Image | None = None,
    dose_label: str | None = None,
    dose_threshold: float | None = None,
    anatomical_roi: sitk.Image | None = None,
) -> Figure:
    """Map direction of perfusion change and its intersection with covered dose."""
    _require_dose_label(dose, dose_label)
    if dose is not None and (dose_threshold is None or not np.isfinite(dose_threshold)):
        raise ValueError("A finite dose threshold is required for candidate dose overlap.")
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), layout="constrained")
    limits = _limits(_data(anatomy))
    for axis, display in zip(axes, _slices(anatomy, _point(anatomy, candidate))):
        _draw(axis, display, display.sample(anatomy), _location(display), limits=limits)
        values = display.sample(signed, categorical=True)
        dose_mask = None
        if dose is not None:
            dose_values = display.sample(dose)
            dose_mask = np.isfinite(dose_values) & (dose_values >= dose_threshold)
        labels = display.sample(candidate, categorical=True)
        _candidate_layers(axis, display, values, np.isfinite(labels) & (labels != 0), dose_mask)
        if anatomical_roi is not None:
            _outline(axis, display, display.sample(anatomical_roi, categorical=True) > 0, "#46dddb", width=1.5)
    legend = [Patch(color=DECREASE, label="Decreased rCBV"), Patch(color=INCREASE, label="Increased rCBV")]
    if dose is not None:
        legend.extend([Patch(color=HIGH_DOSE, label=f"≥ {dose_threshold:g} {dose_label}"),
                       Patch(color=OVERLAP, label="Candidate dose–perfusion overlap")])
    if anatomical_roi is not None:
        legend.append(Patch(color="#46dddb", label="Anatomical ROI boundary"))
    figure.legend(handles=legend, loc="outside lower center", ncol=min(3, len(legend)), frameon=False)
    figure.suptitle("Candidate perfusion change regions")
    return figure


def dose_association_figure(
    dose_array: np.ndarray,
    signed_array: np.ndarray,
    magnitude_array: np.ndarray,
    percent_array: np.ndarray,
    valid_array: np.ndarray,
    bin_edges: Iterable[float],
    *,
    dose_label: str,
) -> Figure:
    """Show descriptive voxel associations and dose-bin change distributions."""
    if len({array.shape for array in (dose_array, signed_array, magnitude_array, percent_array, valid_array)}) != 1:
        raise ValueError("Dose, change maps, and coverage mask must share a shape.")
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    valid = np.asarray(valid_array, dtype=bool) & np.isfinite(dose_array)
    rng = np.random.default_rng(321)
    for axis, values, label in zip(
        axes.ravel()[:3],
        (signed_array, magnitude_array, percent_array),
        ("Signed Δ normalized rCBV", "|Δ normalized rCBV|", "Relative change (%)"),
    ):
        covered = valid & np.isfinite(values)
        dose, change = dose_array[covered], values[covered]
        if len(dose) > 15000:
            selection = rng.choice(len(dose), 15000, replace=False)
            dose, change = dose[selection], change[selection]
        axis.scatter(dose, change, s=4, alpha=0.25, color="#2464a0", edgecolors="none", rasterized=True)
        axis.set_xlabel(dose_label)
        axis.set_ylabel(label)
        axis.axhline(0, color="#9a9a9a", linewidth=0.7)

    covered = valid & np.isfinite(signed_array)
    dose, change = dose_array[covered], signed_array[covered]
    edges = np.asarray(tuple(bin_edges), dtype=float)
    samples, labels = [], []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (dose >= lower) & ((dose <= upper) if index == len(edges) - 2 else (dose < upper))
        if np.any(in_bin):
            samples.append(change[in_bin])
            labels.append(f"{lower:g}–{upper:g}")
    distribution = axes[1, 1]
    if samples:
        distribution.boxplot(samples, showfliers=False, showmeans=True)
        distribution.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
    else:
        distribution.text(0.5, 0.5, "No covered voxels in specified bins", ha="center", transform=distribution.transAxes)
    distribution.set_xlabel(f"Dose interval · {dose_label}")
    distribution.set_ylabel("Signed Δ normalized rCBV")
    figure.suptitle("Descriptive dose–perfusion associations · up to 15,000 voxels per scatter plot")
    return figure


def make_viewer(
    anatomy: sitk.Image,
    may: sitk.Image,
    october: sitk.Image,
    signed: sitk.Image,
    *,
    dose: sitk.Image | None = None,
    dose_label: str | None = None,
    ct: sitk.Image | None = None,
    percent: sitk.Image | None = None,
    change_threshold: float = 0.25,
    dose_threshold: float = 40.0,
    interactive: bool = True,
):
    """Build a synchronized Jupyter slice viewer or a static three-plane figure."""
    _require_dose_label(dose, dose_label)
    if not np.isfinite(change_threshold) or change_threshold < 0:
        raise ValueError("Viewer change threshold must be finite and nonnegative.")
    if not np.isfinite(dose_threshold):
        raise ValueError("Viewer dose threshold must be finite.")

    center = _point(anatomy)
    lower, upper = _bounds(anatomy)
    n_limits = _limits(_data(may), _data(october))
    anatomy_limits = _limits(_data(anatomy))
    ct_limits = _limits(_data(ct)) if ct is not None else anatomy_limits
    map_options = {
        "May": (may, "viridis", n_limits),
        "October": (october, "viridis", n_limits),
        "Signed change": (signed, "RdBu_r", _limits(_data(signed), symmetric=True)),
    }
    if percent is not None:
        map_options["Relative change (%)"] = (percent, "RdBu_r", _limits(_data(percent), symmetric=True))

    def draw(axis, image_slice, name, alpha, show_ct, show_dose, show_candidates, change, dose_min):
        background = ct if show_ct and ct is not None else anatomy
        background_values = image_slice.sample(background)
        limits = ct_limits if show_ct and ct is not None else anatomy_limits
        anatomy_name = "CT" if show_ct and ct is not None else "May MRI"
        title = f"{name} on {anatomy_name}\n{_location(image_slice)}"
        _draw(axis, image_slice, background_values, title, limits=limits)
        image, cmap, limits = map_options[name]
        layer = _layer(axis, image_slice, image_slice.sample(image), cmap, limits, alpha)
        if show_dose and dose is not None:
            dose_plane = image_slice.sample(dose)
            dose_mask = np.isfinite(dose_plane) & (dose_plane >= dose_min)
            _outline(axis, image_slice, dose_mask, HIGH_DOSE, width=1.4)
        if show_candidates:
            change_plane = image_slice.sample(signed)
            _outline(axis, image_slice, (change_plane < 0) & (change_plane <= -change), DECREASE, width=1.7)
            _outline(axis, image_slice, (change_plane > 0) & (change_plane >= change), INCREASE, width=1.7)
        return layer

    def static():
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), layout="constrained")
        for axis, image_slice in zip(axes, _slices(anatomy, center)):
            layer = draw(axis, image_slice, "Signed change", 0.72, False, True, True, change_threshold, dose_threshold)
        _colorbar(figure, layer, axes, "Δ normalized rCBV")
        figure.legend(
            handles=[Patch(color=DECREASE, label="Decreased rCBV"), Patch(color=INCREASE, label="Increased rCBV")],
            loc="outside lower center", ncol=2, frameon=False,
        )
        heading = "Static viewer · candidate contours"
        if dose is not None:
            heading += f" · amber dose contour ≥ {dose_threshold:g} {dose_label}"
        figure.suptitle(heading)
        return figure

    if not interactive:
        return static()

    try:
        import ipywidgets as widgets
    except ImportError:
        return static()

    controls = {
        "plane": widgets.Dropdown(options=tuple(PLANES), description="Plane"),
        "position": widgets.FloatSlider(description="Slice (mm)", continuous_update=False, readout_format=".1f"),
        "map": widgets.Dropdown(options=tuple(map_options), value="Signed change", description="Map"),
        "alpha": widgets.FloatSlider(description="Map opacity", min=0, max=1, value=0.72, step=0.05, continuous_update=False),
        "ct": widgets.Checkbox(description="Show CT", disabled=ct is None, indent=False),
        "dose": widgets.Checkbox(description="Dose contour", value=dose is not None, disabled=dose is None, indent=False),
        "candidates": widgets.Checkbox(description="Change contours", value=True, indent=False),
        "pair": widgets.Checkbox(description="Compare May / October", indent=False),
        "change": widgets.FloatText(description="|Δ rCBV| ≥", value=change_threshold),
        "dose_min": widgets.FloatText(description="Dose ≥", value=dose_threshold, disabled=dose is None),
        "column": widgets.FloatSlider(description="Probe x (mm)", continuous_update=False, readout_format=".1f"),
        "row": widgets.FloatSlider(description="Probe y (mm)", continuous_update=False, readout_format=".1f"),
    }
    probe_text = widgets.HTML()
    plot = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
    updating_plane = False

    def set_range(slider, axis, title):
        # Widen first so a new world coordinate interval can lie outside the old one.
        slider.min = min(float(lower[axis]), slider.min)
        slider.max = max(float(upper[axis]), slider.max)
        slider.value = float(center[axis])
        slider.min, slider.max = float(lower[axis]), float(upper[axis])
        slider.step = max(min(anatomy.GetSpacing()) / 2, 0.05)
        slider.description = f"{title} {'xyz'[axis]} (mm)"

    def update_plane(_change=None):
        nonlocal updating_plane
        updating_plane = True
        plane = PLANES[controls["plane"].value]
        set_range(controls["position"], plane.normal, "Slice")
        set_range(controls["column"], plane.horizontal, "Probe")
        set_range(controls["row"], plane.vertical, "Probe")
        updating_plane = False
        render()

    def probe(image_slice, horizontal, vertical):
        point = [0.0] * 3
        point[image_slice.plane.horizontal] = horizontal
        point[image_slice.plane.vertical] = vertical
        point[image_slice.plane.normal] = image_slice.location
        continuous = anatomy.TransformPhysicalPointToContinuousIndex(point)
        index = tuple(int(np.floor(value + 0.5)) for value in continuous)
        coordinates = ", ".join(f"{coord:.2f}" for coord in point)
        if any(value < 0 or value >= size for value, size in zip(index, anatomy.GetSize())):
            return f"Probe LPS ({coordinates}) mm · outside reference imaging coverage"
        values = {"May": may, "October": october, "Δ rCBV": signed, "Relative Δ (%)": percent, "CT": ct, dose_label: dose}
        voxel = anatomy.TransformIndexToPhysicalPoint(index)
        voxel_location = ", ".join(f"{coord:.2f}" for coord in voxel)
        measurements = []
        for name, image in values.items():
            if image is None:
                continue
            index_for_image = image.TransformPhysicalPointToIndex(voxel)
            if all(0 <= value < size for value, size in zip(index_for_image, image.GetSize())):
                value = float(image.GetPixel(index_for_image))
                measurements.append(f"{name} {value:.4g}" if np.isfinite(value) else f"{name} unavailable")
        return f"Probe LPS ({coordinates}) mm · reference voxel {index} at ({voxel_location}) mm · " + " · ".join(measurements)

    def render(_change=None):
        if updating_plane:
            return
        values = {name: widget.value for name, widget in controls.items()}
        if values["change"] < 0 or not np.isfinite(values["change"]) or not np.isfinite(values["dose_min"]):
            probe_text.value = "Enter finite thresholds and use a nonnegative change threshold."
            return
        image_slice = _slice(anatomy, PLANES[values["plane"]], values["position"])
        names = ("May", "October") if values["pair"] else (values["map"],)
        figure, axes = plt.subplots(1, len(names), squeeze=False, figsize=(5.4 * len(names), 4.8),
                                    layout="constrained")
        try:
            for axis, name in zip(axes[0], names):
                layer = draw(axis, image_slice, name, values["alpha"], values["ct"], values["dose"],
                             values["candidates"], values["change"], values["dose_min"])
                axis.axvline(values["column"], color="#ffffff", linestyle=":", linewidth=0.8)
                axis.axhline(values["row"], color="#ffffff", linestyle=":", linewidth=0.8)
                labels = {"Signed change": "Δ normalized rCBV", "Relative change (%)": "Relative change (%)"}
                _colorbar(figure, layer, axis, labels.get(name, "Normalized rCBV"))
            if values["dose"]:
                figure.suptitle(f"Amber contour · ≥ {values['dose_min']:g} {dose_label}")
            if values["candidates"]:
                figure.legend(handles=[
                    Patch(color=DECREASE, label="Decreased rCBV"),
                    Patch(color=INCREASE, label="Increased rCBV"),
                ], loc="outside lower center", ncol=2, frameon=False)
            with BytesIO() as buffer:
                figure.savefig(buffer, format="png", dpi=105, facecolor="white")
                plot.value = buffer.getvalue()
        finally:
            plt.close(figure)
        probe_text.value = (
            f"{escape(probe(image_slice, values['column'], values['row']))}<br>"
            "Change contours use signed change in paired coverage at the viewer threshold."
        )

    controls["plane"].observe(update_plane, names="value")
    for name, widget in controls.items():
        if name != "plane":
            widget.observe(render, names="value")
    update_plane()
    row_layout = widgets.Layout(flex_flow="row wrap")
    return widgets.VBox([
        widgets.HBox([controls["plane"], controls["position"], controls["pair"]], layout=row_layout),
        widgets.HBox([controls["map"], controls["alpha"], controls["ct"]], layout=row_layout),
        widgets.HBox([controls["candidates"], controls["change"], controls["dose"], controls["dose_min"]], layout=row_layout),
        widgets.HBox([controls["column"], controls["row"]], layout=row_layout),
        probe_text,
        plot,
    ])
