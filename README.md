# Longitudinal rCBV and radiation dose analysis

A research workflow for measuring voxelwise change in leakage-corrected relative cerebral blood volume (rCBV) between baseline and follow-up MRI examinations and examining its spatial association with an accumulated radiation dose distribution.

## Install and run the synthetic example

From the repository root, create a Python 3.11 environment or use a compatible Python 3.12 environment and run the synthetic example.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run_analysis.py --synthetic --output outputs/synthetic
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m jupyter lab
```

### Linux or macOS

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python run_analysis.py --synthetic --output outputs/synthetic
.venv/bin/python -m pytest -q
.venv/bin/python -m jupyter lab
```

Open [Longitudinal_rCBV_EQD2_Analysis.ipynb](notebooks/Longitudinal_rCBV_EQD2_Analysis.ipynb) with the same environment's Python kernel. Run all cells to generate a new synthetic DICOM example and its report. The viewer includes interactive slice and overlay controls in a widget-enabled Jupyter frontend, with static figures for headless execution.

## Analyze local DICOM data on Windows

Open PowerShell using the Windows account that can read your DICOM folders. Replace the example path with the location of your local data and verify that it is accessible.

```powershell
Test-Path -LiteralPath 'C:\path\to\dicom'
```

1. Copy [patient_config.yaml](config/patient_config.yaml) to `config/local_patient.yaml`. Set the four source directory paths under `paths` for the baseline rCBV, follow-up rCBV, planning CT, and RTDOSE. Relative input and dose transform paths resolve from the directory containing the YAML. Keep source DICOM folders read-only and point the output argument to a separate directory. Keep patient masks, screenshots, reports and outputs local.
2. Run inspection to list series, geometry, metadata and warnings. Review May and October study candidates for coverage, quality, and sequence suitability. Enter the chosen series under `series_uids.may_anatomy` and `series_uids.october_anatomy` or set their directories under `paths`. Record the selection rationale under `anatomy_selection_reason.may` and `anatomy_selection_reason.october`. If several candidates match, inspection writes the inventory and asks for a series selection. Check the acquisition dates, rCBV numerical mapping and dose references.

   ```powershell
   .\.venv\Scripts\python.exe run_analysis.py --config config/local_patient.yaml --output outputs/patient --stage inspect
   ```

3. Supply reviewed brain and normal-appearing white matter masks in the May reference frame at `paths.brain_mask` and `paths.reference_roi`. Record the rCBV value mapping and its source in `rcbv.quantitative_evidence`, the ROI selection rationale in `normalization.reference_description`, and the within-session alignment policy under `registration`. Finite negative rCBV values are retained by default. Set `rcbv.exclude_negative` only if source documentation identifies those values as invalid. Supply a verified CT-to-dose transform and `dose.spatial_evidence` if dose and CT use different frame UIDs. Run the registration stage and review the QA outputs across all three planes.

   ```powershell
   .\.venv\Scripts\python.exe run_analysis.py --config config/local_patient.yaml --output outputs/patient --stage register
   ```

4. Review the rCBV-to-anatomy alignment at both dates and the dose-to-CT correspondence. For each confirmed shared-frame alignment, set the corresponding `registration.same_frame_rcbv` entry (`may` or `october`) to `true`. If the alignment requires correction, select `rigid` under that session's `registration.rcbv_method` and rerun registration before recording approval. After accepting registration, copy the fingerprint from `quantitative_results/registration.json` into the following fields in your YAML. Enter your reviewer name and actual review findings. Use `rejected` when the alignment is unacceptable. Changes to registration inputs can require a new review.

   ```yaml
   registration:
     review:
       status: approved
       reviewer: "Reviewer name"
       notes: "Describe the review and supporting evidence"
       fingerprint: "Paste the fingerprint from registration.json"
   ```

   Optional numerical QA thresholds can be set directly under `registration.thresholds` as shared settings. A nested `ct`, `october`, `may_rcbv`, or `october_rcbv` mapping overrides those shared settings for that pair. For example, use `min_fixed_coverage` to flag insufficient overlap and `max_tre_mm` when corresponding landmarks are available. Set limits based on the images and intended analysis.

5. Run the full analysis using the reviewed configuration. To use the notebook on the same workstation, set its configuration and output options before launching Jupyter.

   ```powershell
   .\.venv\Scripts\python.exe run_analysis.py --config config/local_patient.yaml --output outputs/patient
   $env:NECROSIS_CONFIG = (Resolve-Path config/local_patient.yaml).Path
   $env:NECROSIS_OUTPUT = Join-Path (Get-Location) 'outputs/patient_notebook'
   .\.venv\Scripts\python.exe -m jupyter lab
   ```

Use the same phase commands with the path to your local configuration if you keep it outside the checkout. A Jupyter server launched on a remote machine needs its own access to the source data. Run locally when the mapped drive is visible only to your workstation.

## Registration and measurements

| Step | Method |
| --- | --- |
| DICOM inspection | Inventory by study and series, reconstruct slices using image positions projected onto the slice normal, and review spacing, orientation, frame and intensity scaling. |
| Registration | Six-degree-of-freedom rigid CT and October MRI registration to May anatomy. Check each rCBV-to-anatomy mapping separately. SimpleITK resampling takes a transform from reference physical points to sampling points in the source image. |
| Geometry and coverage | DICOM and SimpleITK use LPS millimetres. NIfTI exports carry the corresponding RAS affine. NumPy volumes are indexed `[z, y, x]`. Compose sampling transforms before mapping quantitative images to the May grid. Use continuous interpolation for scalar images and nearest neighbour for labels. |
| Registration QA | Review before/after fusion, checkerboards, boundaries and overlap alongside metrics, independent masks or landmarks when supplied. Inspect local alignment near the region under study and record a decision. |
| Normalization | Use one anatomically consistent normal-appearing white matter ROI, preferably contralateral to the treated lesion and within shared valid perfusion coverage. Compare the raw and normalized distributions. Per-examination ROI medians can absorb true global changes. Confirm whether the maps are already comparable before deciding to normalize again. |
| Change analysis | Calculate October minus May, magnitude, percent change with a baseline floor, and directional change. Restrict calculations to common brain coverage. Calculate candidate volumes in mL and examine threshold sensitivity. |
| Dose analysis | Apply `DoseGridScaling` in the dataset's stated units, resolve the dose-to-CT relationship, then sample dose in the May frame. Use only jointly valid dose and perfusion coverage for correlations and overlap denominators. |
| Export | Review registered images and validity maps, QA and publication figures, quantitative CSV/JSON exports, and the readable analysis report in the selected output directory. |

Skull, CSF, vessels, lesion, treatment changes, and susceptibility artefact are poor reference tissue candidates. Select the ROI on reviewed anatomical images and inspect the same ROI on both perfusion maps. The percent-change map excludes small baseline values because their ratios are unstable.

A shared frame UID supplies a DICOM-coordinate relationship. Review the actual spatial alignment for motion, distortion and incomplete coverage. In dose QA, confirm the applicable plans, frame and accumulation reference. A directory name or `DoseUnits = GY` alone cannot establish EQD2. Set `dose.interpretation: eqd2` and document `dose.evidence` when provenance supports that label. Record the evidence, alpha/beta ratio, accumulation method, and treatment dates, and check treatment timing relative to both MRI examinations. Dose interpolation keeps the source scale and carries no biological conversion.

Decreased and increased rCBV have several possible explanations, including mixed injury and tumour, artefact and acquisition differences. A high-dose overlap is a candidate dose-associated perfusion finding. Voxel associations are descriptive within a single case because neighbouring observations are spatially correlated. Clinical follow-up and, when available, pathology can help interpret those findings.

## Validation scope

The synthetic dataset provides known geometry, rigid motion, perfusion changes and dose values to test the workflow. The executed notebook recovered the CT and October MRI mappings with maximum landmark errors of 0.136 mm and 0.144 mm, respectively, under the 0.75 mm synthetic tolerance. The May and October white-matter normalization factors were 2.416 and 3.625. The automated tests cover geometry, registration, dose scaling, normalization, change maps and dose statistics. Registration unit tests require landmark errors below 0.45 mm for the correlation cases and 1.0 mm for the multimodal case, with angle tolerances of 0.6° and 1.5°, respectively. An independently constructed oblique RTDOSE case checks coordinates to 10⁻⁸ mm and scaled voxel values to 10⁻⁶ Gy.

To execute the notebook headlessly from the repository root, use the activated project environment. The executed notebook and synthetic results go to `outputs/`.

```bash
NECROSIS_STATIC_VIEWER=1 jupyter nbconvert --to notebook --execute notebooks/Longitudinal_rCBV_EQD2_Analysis.ipynb --output-dir=outputs --ExecutePreprocessor.timeout=600
```

Clinical registration, source-image scaling, dose provenance and patient findings require local data access and review.
