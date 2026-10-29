# Local case pipeline

`run_case_pipeline.py` runs the trajectory-feature and candidate-planning pipeline on **one CT case stored on your own machine**. Nothing is uploaded. It is a research prototype, not a medical device, and its outputs must not be used for clinical decisions.

```text
CT + vertebra mask + baseline screw (entry, tip)
  -> per-vertebra anatomical frame (falls back to image axes if unreliable)
  -> whole vertebral-body trabecular density (patient-intrinsic features)
  -> pedicle isthmus detection and trajectory match
  -> baseline trajectory bone-density features
  -> candidate grid, safety filtering, optional risk scoring
```

## Data handling

- Keep cases **outside this repository**, or in the git-ignored `cases/` and `case_output/` folders. Imaging files (`*.nii*`, `*.dcm`, …) are also git-ignored, but CSVs are not.
- Use de-identified data, within your ethics approval. Converting DICOM to NIfTI (e.g. `dcm2niix`) drops most header metadata.
- The script records image geometry and options in `case_summary.json`, but **not** input paths or DICOM header fields.

## Inputs

| Argument | Content |
| --- | --- |
| `--ct` | CT in HU: NIfTI / NRRD / MHA file, or a DICOM series folder. Thin slices (≤ 1 mm) and a bone or standard kernel are recommended. **CT only** — MRI has no HU scale. |
| `--mask` | Vertebra segmentation: a binary mask of one vertebra (body **and** posterior elements), or a label map plus `--label` (e.g. TotalSegmentator output). Same physical space as the CT; the grid may differ. |
| `--screws` | CSV of baseline (surgeon) screws; see below. |
| `--coords` | Coordinate convention of the CSV points: `lps` (default, SimpleITK/DICOM), `ras` (3D Slicer, most navigation exports), or `voxel` (CT voxel indices). |

### Screw CSV

See `screws_template.csv`.

| Column | Required | Meaning |
| --- | --- | --- |
| `screw_id` | yes | unique name, e.g. `L4_left` |
| `entry_x/y/z` | yes | entry point on the bone surface (mm) |
| `tip_x/y/z` | yes | screw tip, or any point further along the axis when `length` is given |
| `diameter` | yes | outer screw diameter (mm) |
| `length` | no | screw length (mm); default is the entry–tip distance |
| `label` | no | label of the vertebra in the label map; overrides `--label` |

In 3D Slicer, place two fiducials per screw (entry and tip), then copy their RAS coordinates and run with `--coords ras`.

## Options

| Option | Effect |
| --- | --- |
| `--risk none` (default) | features and safety feasibility only; no candidate is chosen |
| `--risk demo` | **placeholder** score for exercising the pipeline; not a loosening probability |
| `--risk my_model.py:predict` | your model: a function taking the feature dict and returning a risk (lower is better) |
| `--pivot entry` / `isthmus` | angle-change pivot (default from the grid: `entry`) |
| `--grid grid_example.json` | override the candidate grid (`CandidateGridConfig` fields) |
| `--no-snap` | do not move candidate entries onto the bone surface |
| `--calibration slope,intercept` | linear HU→BMD calibration (e.g. phantom) |
| `--keep-unreliable-frame` | use the estimated vertebra frame even when its symmetry score is low |
| `--all-features` | write every feature (≈ 500 per candidate) to the candidate CSVs |
| `--no-qc` | skip QC figures (also skipped automatically without matplotlib) |

## Outputs (in `--out`)

| File | Content |
| --- | --- |
| `case_summary.json` | vertebra frame (axes, symmetry score, rotation vs. image axes), vertebral-body BMD, per-screw isthmus geometry, baseline key features and the optimization summary; errors per screw |
| `candidates_<screw>.csv` | one row per candidate: adjustments, final entry/tip, snap status, feasibility and violated constraints, risk, key features |
| `features_<screw>_baseline.json` | all features of the baseline screw |
| `qc_<screw>.png` | axial and sagittal image-index slices through the isthmus with the mask contour, the baseline screw (white) and the selected candidate (orange) |

## What to check on a first real case

1. **QC figures**: the mask contour should follow the vertebra and the white line should run through the pedicle. The figure shows image-index planes; for oblique acquisitions the screw is a projection.
2. **Vertebra frame**: `reliable` and `symmetry_score`, and whether `angles_to_image_axes` looks plausible for the patient (axial rotation, tilts in degrees).
3. **Isthmus**: `outer_min_width_mm` and `inner_min_width_mm` against a manual measurement, and `inner_dominance` (≥ 0.7 expected inside the pedicle). The inner width depends on the 800 HU cortical threshold and on partial-volume effects; it is the most likely value to need calibration.
4. **Vertebral body BMD**: `vertebra_midbody_mean` against a manually placed mid-vertebral ROI.
5. **Candidates**: large `entry_snap_shift_mm` values mean the shifted entry left the lamina and the axis first meets bone deeper; such candidates are effectively a different entry point.

Typical run time is 10–20 s per case on a laptop CPU.
