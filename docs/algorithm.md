# Algorithm Notes

This document describes the behavior implemented in the public source. It does not provide clinical validation or claim that the configured heuristics are scientifically optimal.

## 1. HU Preprocessing

The imaging utilities clip CT intensity values to configured lower and upper bounds and normalize the clipped result to a bounded numeric range. They also provide local-neighborhood extraction and resampling helpers for SimpleITK images.

## 2. Coordinate Transforms

Voxel-to-world and world-to-voxel conversion use SimpleITK image origin, spacing, and direction metadata. The transforms support synthetic image volumes with non-default metadata and are tested for round-trip consistency.

## 3. Geometry Functions

The geometry layer provides vector normalization, distances between points and geometric objects, projections, surface-normal queries, bounds, surface sampling, local coordinate construction, and mesh containment/closest-point helpers. PyVista and VTK supply the mesh representation and spatial operations.

These utilities operate on caller-provided geometry. The repository tests them using synthetic spheres, boxes, planes, and arrays rather than patient-derived models.

## 4. Candidate Generation and Scoring

Candidate entry points are generated in a local geometric neighborhood and evaluated with a weighted heuristic score. The current implementation combines the following terms:

| Term | Configured weight |
| --- | ---: |
| HU | 0.35 |
| Cortical distance | 0.25 |
| Stability | 0.20 |
| Smoothness | 0.10 |
| Anatomical center | 0.10 |

These numbers document the source configuration only. The public package does not attach independent biological or clinical evidence to the selected values.

The candidate workflow uses configured sampling and search parameters, including 20 candidates, selection of the top 5 candidates, 500 surface samples, a local sphere radius of 5, a search radius of 10, and a search step of 2. Units follow the coordinate system of the supplied data.

## 5. Local-Search Optimizer

The optimizer refines a candidate through bounded local search and supports gradient-style and multi-start helpers. The current configuration uses a radius of 3, a step of 0.5, and 5 iterations. The implementation is heuristic and deterministic only to the extent allowed by its inputs and sampling behavior.

The synthetic optimizer test uses an analytic objective with a known optimum. This verifies basic movement toward the expected solution without altering the production search rule.

## 6. Screw-Trajectory Bone-Density Features

`pedicle_planning.ai.trajectory` evaluates a fully specified screw rather than an entry point. A `ScrewPlan` holds the entry point, direction, length, and diameter in the image's physical coordinate system.

- **Sampling.** Points are placed on a uniform lattice (default 0.5 mm axially and in cross-section) inside the screw cylinder (`core`, radius ≤ D/2) and in a surrounding bone–screw interface annulus (`interface`, default 1 mm thick). A separate spherical region around the tip (default radius 4 mm) is sampled as `tip_sphere`.
- **Interpolation.** Physical points are mapped to continuous voxel indices using origin, spacing, *and* the direction matrix, then sampled with trilinear interpolation. Points outside the image are reported as missing rather than padded with zeros.
- **Segments.** The axis is split into `entry`, `pedicle`, `transition`, `body`, and `tip` segments. Defaults are length fractions (0, 0.2, 0.45, 0.6, 0.8, 1.0); explicit millimetre boundaries can be passed, e.g. from a future pedicle-isthmus detector.
- **Bone membership.** An optional binary bone mask (nearest-neighbour lookup) or closed bone mesh restricts statistics to bone and yields the in-bone (purchase) length along the screw axis.
- **Statistics.** For every segment × region: sample counts, bone fraction, mean, median, standard deviation, 10th/90th percentiles, coefficient of variation, low-density fraction (HU < 110 by default), and cortical fraction (HU ≥ 800 by default).
- **Directional half-rings.** When an anatomical frame (medial and cranial directions) is passed, the interface annulus is additionally split into medial/lateral and cranial/caudal 180° halves, using the frame directions projected perpendicular to the screw axis (`interface_medial`, `interface_lateral`, `interface_cranial`, `interface_caudal` for every segment). This raises the flat feature count from 214 to 478; `optimize_screw_plan` passes the screw's frame automatically.
- **Calibration.** Values are HU unless a `LinearCalibration` (e.g. from two phantom inserts) is supplied, in which case density statistics are reported in the calibrated unit; the threshold fractions still use HU.

`TrajectoryFeatures.to_flat_dict()` flattens the result into named scalar features for downstream risk models. Default segment fractions and thresholds are engineering defaults, not anatomically or clinically validated values. This module is additive: it does not change the existing scorer or optimizer.

## 7. Pedicle Isthmus Detection

`pedicle_planning.ai.pedicle.detect_pedicle_isthmus` uses a baseline screw that passes through the pedicle (typically the surgeon's plan), a binary vertebra mask, optionally the CT, and the screw's anatomical frame (medial and cranial directions).

- **Ray sections.** Every 1 mm over 5–70 % of the screw length, a plane perpendicular to the screw axis is probed with rays from the axis point every 15° (0° medial, 90° cranial, 180° lateral, 270° caudal), sampled every 0.25 mm up to 15 mm. The outer wall is where a ray leaves the mask for at least 0.75 mm. Transverse width = medial + lateral distance and height = cranial + caudal distance, each side the median of three neighbouring rays. A process attached to the pedicle (transverse process, superior articular process) only lengthens rays in its direction; it does not invalidate the section.
- **Cortex.** With CT, each section sets its own threshold halfway between the cancellous level (median HU along the inner half of the rays) and the cortical peak (median of the per-ray maxima within 4 mm inside the outer wall). The inner wall is where a ray first crosses that threshold within those 4 mm. If the contrast is below 100 HU the cortex is reported as unresolved and inner = outer, instead of applying a fixed HU threshold that thin, partial-volume-blurred cortex may never reach.
- **Isthmus and pedicle segment.** The isthmus is the section with the smallest transverse width (3-section median) among sections whose medial and lateral rays both close. The pedicle segment is the contiguous range with width ≤ 1.3× the isthmus width.
- **Pedicle axis.** The medial–lateral and cranial–caudal offsets of the section chord centres from the screw axis are regressed on axial position with a Theil–Sen (median pairwise) slope; the slopes tilt the screw direction into the pedicle axis. With fewer than 4 sections or a tilt above 30° the screw direction is kept. Principal-component fits of the few, noisy centres in a short pedicle proved unstable. The isthmus is then re-measured perpendicular to the fitted axis (accepted only if it does not shrink below 60 %).
- **Trajectory match.** For any screw: angle to the pedicle axis, offset from the isthmus centre, and, from rays cast in the isthmus plane at the screw's crossing point, the clearance of the screw surface to the outer wall, to the inner cortical wall, and separately to the medial and lateral walls; diameter-to-inner-width ratio.

## 8. Candidate Generation and Risk-Based Selection

`pedicle_planning.ai.candidates` starts from the baseline plan and an `AnatomicalFrame` (medial and cranial directions for that screw).

- **Grid.** By default: entry shifted −2…+2 mm medially in 1 mm steps, medial angulation −5/0/+5°, length −5/0/+5 mm, and diameter +0/+0.5 mm (90 candidates including the baseline). Cranial shifts and cranio-caudal angulation are available but default to no change; absolute length and diameter catalogues can be passed instead of offsets.
- **Pivot.** Angle changes rotate either about the (shifted) entry point (`PIVOT="entry"`, default) or about the point where the baseline axis crosses the isthmus plane (`PIVOT="isthmus"`). With the isthmus pivot every rotated axis still passes through the (shifted) isthmus point and the entry point moves instead, mirroring how the pedicle acts as the channel during insertion. Medial/cranial shifts translate the whole axis in both modes.
- **Entry on bone surface.** With a bone mask or mesh, each non-baseline candidate's entry point is moved along its new axis to the nearest outside-to-inside bone crossing within −10…+10 mm (`SNAP_ENTRY`); length is measured from that point, so the tip moves with it. Candidates without a crossing are marked `failed` and rejected by `entry_on_bone_surface`. The baseline plan is kept unchanged as the reference. The summary reports the resulting entry displacement in medial, cranial, and axial components.
- **Safety filter.** Built-in hard constraints: entry reachable on the bone surface, diameter ≤ isthmus inner width − 1 mm, non-negative pedicle-wall clearance at the isthmus, and a 2 mm in-bone margin around the tip. Constraints whose data are absent are skipped. Spinal-canal, nerve-root, endplate, and facet constraints require additional segmentation and can be added as callables.
- **Scoring.** For each feasible candidate the trajectory features (Section 6, with anatomical segment bounds) and `match_*` features are recomputed and passed to a caller-supplied `risk_fn(features) -> float`. The lowest-risk feasible candidate is selected; ties keep the baseline.
- **Output.** `OptimizationResult.summary()` reports baseline and best risk, absolute and relative risk reduction, entry offsets, angle changes, and the recommended length and diameter.

The package contains no trained loosening-risk model; tests use a synthetic vertebra and a toy risk function.

## 9. Per-Vertebra Anatomical Frame

`pedicle_planning.ai.vertebra_frame.estimate_vertebra_frame` derives the medial and cranial directions used by the candidate grid from a single-vertebra mask, so that axially rotated or tilted vertebrae (e.g. in scoliosis) are handled.

- **Left–right axis.** The midsagittal plane through the mask centroid is found by maximizing mirror overlap of the mask (coarse-to-fine search, ±40° in axial rotation and coronal tilt around the image axes).
- **Cranial axis.** Within that plane, the anterior 40 % of the vertebra (the body, excluding lamina and spinous process) is projected, and the direction of minimum 1st–99th percentile width (body height) is taken within ±30° of the initial direction. Using percentiles rather than variance avoids bias from asymmetric posterior structures; the trimmed range keeps small osteophytes from tilting the axis.
- **Signs and origin.** Image patient directions (LPS by default) only fix axis signs and search start; the frame is right-handed (left = cranial × anterior). The origin is the body centroid projected onto the midsagittal plane.
- **Per-screw frame.** `anatomical_frame_for(entry)` picks the side from the entry point and returns medial pointing to the midline.
- **Quality flag.** `symmetry_score` is the mirror-overlap fraction; `reliable` is false below 0.85 (heuristic, to be calibrated on real segmentations). `image_axes_frame` is the fallback using image axes directly.

On the synthetic vertebra, axes are recovered within about 1–2° for combined rotations up to 35° axial, 12° coronal, and 20° sagittal, including with an osteophyte-like spur or missing posterior elements on one side.

## 10. Whole Vertebral-Body Bone Density

`pedicle_planning.ai.vertebral_body.compute_vertebral_body_bmd` gives patient-intrinsic density features that do not change with the screw plan.

- **Body region.** In the per-vertebra frame (Section 9), the anterior 40 % of the vertebra along the anteroposterior axis is taken as the vertebral body, excluding pedicles, lamina, and spinous process.
- **Trabecular region.** The body is eroded by 3 mm to remove the cortical shell and endplates (`trabecular`).
- **Mid-body slab.** A 10 mm slab centred at mid-height of the body (1st–99th percentile extent along the cranial axis) approximates the clinical mid-vertebral cancellous ROI (`midbody`).
- **Output.** The same statistics as the trajectory regions, plus body and trabecular volume and body height; optional linear HU→BMD calibration. `to_flat_dict()` results can be passed to `optimize_screw_plan(patient_features=...)`, which merges them into every candidate's features (key clashes are rejected).

The ROI is fully automatic and three-dimensional, so it is not identical to manually placed single-slice clinical ROIs; values should be compared with those only after validation.

## 11. Tetrahedral Transition Algorithms

The FEA package contains transition templates for tetrahedral refinement and utilities that validate generated tetrahedra. Validation checks include element volume and orientation, duplicate elements, non-manifold faces, interface signatures, and elementary mesh-quality measures.

Additional helpers group green-transition faces and triangulate shared interfaces consistently. The templates represent computational mesh preprocessing; they are not a complete adaptive meshing system or a claim of solver accuracy.

## 12. Abaqus Export

The export layer writes nodes and four-node tetrahedral elements as Abaqus C3D4 part definitions and assembles generated parts into an Abaqus input structure. It can parse the relevant mesh text used by the writer and reports basic counts and status.

The repository does not bundle Abaqus, invoke an Abaqus solver, provide material calibration, or include patient-specific input/output files. Generated syntax is covered with minimal programmatically constructed tetrahedral examples.
