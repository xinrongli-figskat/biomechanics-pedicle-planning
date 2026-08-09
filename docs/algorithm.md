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

## 6. Tetrahedral Transition Algorithms

The FEA package contains transition templates for tetrahedral refinement and utilities that validate generated tetrahedra. Validation checks include element volume and orientation, duplicate elements, non-manifold faces, interface signatures, and elementary mesh-quality measures.

Additional helpers group green-transition faces and triangulate shared interfaces consistently. The templates represent computational mesh preprocessing; they are not a complete adaptive meshing system or a claim of solver accuracy.

## 7. Abaqus Export

The export layer writes nodes and four-node tetrahedral elements as Abaqus C3D4 part definitions and assembles generated parts into an Abaqus input structure. It can parse the relevant mesh text used by the writer and reports basic counts and status.

The repository does not bundle Abaqus, invoke an Abaqus solver, provide material calibration, or include patient-specific input/output files. Generated syntax is covered with minimal programmatically constructed tetrahedral examples.
