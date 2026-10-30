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

## Check the inputs first

```bash
python examples/clinical/inspect_inputs.py --ct cases/case01/ct.nii --mask cases/case01/seg/vertebrae_L4.nii.gz
```

This prints size, spacing, orientation, intensity percentiles and mask labels, and warns when the CT does not look like HU (e.g. a converter dropped the DICOM rescale intercept), when slices are thick, or when the mask is not in the same space as the CT. Copy cases to a short path without spaces or non-ASCII characters (e.g. `D:\cases\case01\`), which avoids path problems in SimpleITK and TotalSegmentator on Windows.

## Starting from planning-system meshes

If a planning system stored the segmented bone as a surface mesh (e.g. STL) and the planned screws as positioned screw models, convert them first:

```bash
python examples/clinical/prepare_from_meshes.py --ct cases/case01/ct.nii --bone-mesh cases/case01/bone.stl --screw-mesh cases/case01/screw_a.stl --out cases/case01/prepared
```

- Each mesh is matched to the CT in LPS or RAS automatically (`--mesh-coords` to force); meshes that overlap the CT in neither (unpositioned templates) are skipped.
- Bone meshes are rasterized on the CT grid; separate vertebrae (e.g. L3 and L4 in one STL) become separate labels, numbered cranial to caudal (1 = most cranial). Compare `rasterized_volume_ml` with `mesh_volume_ml`: a large gap means the mesh is not closed. A single lumbar vertebra is roughly 50–90 ml; if two vertebrae come out as one label because the surfaces touch, pass `--vertebrae 2` to split them (erosion + watershed).
- Each screw's axis comes from the mesh shape. The radius profile along the axis identifies a wider end segment (screw head / tulip), which is excluded: `length_mm` and `diameter_mm` describe the threaded shank and the entry is the shank end next to the head. Without a head, the entry is the end with less bone around it. The label is the vertebra the axis runs through.
- Outputs (`vertebrae_from_mesh.nii.gz`, `screws_from_mesh.csv` with a `label` column, `prepare_report.json`) use neutral names (`screw1`, `label 1`), never input file names. Pass them to `run_case_pipeline.py` with `--mask` and `--screws` (LPS, the default).

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
