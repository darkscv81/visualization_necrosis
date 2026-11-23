"""Serialize local research outputs with explicit missing-value handling."""

import csv
import json
from pathlib import Path

import numpy as np


def serializable(value):
    """Convert NumPy values and non-finite numbers to strict JSON equivalents."""
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(data), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(serializable(rows))
    return path


def write_report(path: Path, report: dict, figures: list[Path]) -> Path:
    """Write a human-readable local summary with links to detailed exports."""
    lines = [
        "# Longitudinal perfusion and radiation dose",
        "",
        f"Dataset mode: **{report['mode']}**",
        f"Registration review: **{report['registration']['review']['status']}**",
        f"Dose quantity: **{report['dose_label']}**",
        "",
        "## Imaging and geometry",
        "",
        "Spatial computations use DICOM LPS millimetres. NIfTI exports use the "
        "corresponding affine written by SimpleITK.",
        "",
        "| Dataset | Acquisition or study date | Size (x, y, z) | Spacing (mm) |",
        "|---|---|---|---|",
    ]
    for name, info in report["inspection"]["images"].items():
        geometry = info["pipeline_geometry"]
        date = info.get("pipeline_date") or "Unknown"
        size = " × ".join(map(str, geometry["size_xyz"]))
        spacing = " × ".join(f"{x:.3g}" for x in geometry["spacing_mm_xyz"])
        lines.append(f"| {name} | {date} | {size} | {spacing} |")
    lines.extend([
        "",
        "Series selection, coordinate and scaling metadata, registered field-of-view "
        "coverage, transform parameters, metric values, and review details are "
        "included in the [quantitative JSON](../quantitative_results/analysis.json).",
        "",
        "## Perfusion and dose results",
        "",
        "Normalization:",
        "",
        "```json",
        json.dumps(serializable(report["normalization"]), indent=2),
        "```",
        "",
        "Change and dose summaries:",
        "",
        "```json",
        json.dumps(serializable({
            "change": report["change"],
            "dose": report["dose"]["summary"],
            "dose_provenance": report["registration"]["dose_provenance"],
            "associations": report["dose"]["associations"],
        }), indent=2),
        "```",
        "",
        "## Figures",
        "",
    ])
    lines.extend(f"- [{figure.stem.replace('_', ' ')}](../{figure.parent.name}/{figure.name})"
                 for figure in figures)
    lines.extend(["", "## Interpretation and limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
