# Limitations and Intended Use

## Research Scope

This repository is a computational research prototype and portfolio artifact. It is not a medical device and is not intended for diagnosis, treatment planning, surgical navigation, or other clinical decision making.

## Synthetic Testing

Automated tests use programmatically generated NumPy arrays, SimpleITK metadata, elementary PyVista/VTK geometry, and minimal tetrahedral meshes. This is useful for checking software behavior and numerical invariants, but it does not measure performance on clinical anatomy.

No real patient DICOM or NIfTI images, STL anatomy, patient metadata, or patient-derived results are included.

## Data Limitations

- No public clinical benchmark is evaluated here.
- No variability across scanners, acquisition protocols, reconstruction kernels, anatomy, pathology, or implants is characterized.
- HU behavior is demonstrated computationally; it is not calibrated here against a specific scanner or material standard.
- Synthetic primitives do not reproduce the full geometric complexity or uncertainty of vertebral anatomy.

## Algorithm Assumptions

- Entry-point planning is a manually weighted heuristic based on CT intensity and geometric features.
- Candidate sampling and local search can depend on initialization, bounds, mesh quality, coordinate conventions, and configured search resolution.
- Local optimization does not guarantee a global optimum.
- Score weights and search parameters are retained from the current implementation and are not presented as universally valid or clinically optimal.
- Geometric safety calculations depend on the correctness and resolution of caller-provided meshes and metadata.

## No Trained ML Model

The public package does not contain a trained neural network, learned parameters, a training pipeline, or model-performance claims. The term `ai` is retained as a package namespace inherited from the research prototype; the currently published planning method is heuristic.

## Clinical Validation

There has been no prospective clinical validation of this public package. The repository does not report surgical outcomes, diagnostic accuracy, regulatory clearance, or clinical equivalence. Results from synthetic unit tests must not be interpreted as evidence of clinical safety or effectiveness.

## FEA Scope

- The FEA code covers selected mesh-transition, validation, and Abaqus text-export components.
- It does not provide a complete patient-specific preprocessing pipeline.
- It does not define or validate a complete set of material properties, contacts, loads, or boundary conditions.
- It does not run a solver or validate solver convergence.
- Exported meshes and assemblies are computational artifacts, not validated biomechanical or clinical predictors.

## Excluded Components

The public scope excludes clinical data ingestion, segmentation, patient databases, the original GUI, bundled runtimes, third-party binaries, unreviewed research modules, screw CAD, and complete experiment/result management.
