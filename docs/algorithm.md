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
- **Calibration.** Values are HU unless a `LinearCalibration` (e.g. from two phantom inserts) is supplied, in which case density statistics are reported in the calibrated unit; the threshold fractions still use HU.

`TrajectoryFeatures.to_flat_dict()` flattens the result into named scalar features for downstream risk models. Default segment fractions and thresholds are engineering defaults, not anatomically or clinically validated values. This module is additive: it does not change the existing scorer or optimizer.

## 7. Tetrahedral Transition Algorithms

The FEA package contains transition templates for tetrahedral refinement and utilities that validate generated tetrahedra. Validation checks include element volume and orientation, duplicate elements, non-manifold faces, interface signatures, and elementary mesh-quality measures.

Additional helpers group green-transition faces and triangulate shared interfaces consistently. The templates represent computational mesh preprocessing; they are not a complete adaptive meshing system or a claim of solver accuracy.

## 8. Abaqus Export

The export layer writes nodes and four-node tetrahedral elements as Abaqus C3D4 part definitions and assembles generated parts into an Abaqus input structure. It can parse the relevant mesh text used by the writer and reports basic counts and status.

The repository does not bundle Abaqus, invoke an Abaqus solver, provide material calibration, or include patient-specific input/output files. Generated syntax is covered with minimal programmatically constructed tetrahedral examples.
