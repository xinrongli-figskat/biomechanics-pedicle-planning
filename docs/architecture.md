# Architecture

## Public Package Boundary

The public package has two top-level computational areas:

- `pedicle_planning.ai`: CT preprocessing, coordinates, geometry, candidate scoring, and heuristic optimization.
- `pedicle_planning.fea`: tetrahedral transition utilities, mesh validation, and Abaqus C3D4 export.

They are deliberately independent at the Python import level. Neither package imports a GUI, a patient database, a bundled runtime, or modules from the original research project.

## AI Module Dependencies

```text
pedicle_planning.ai
├── inference
│   └── predictor
│       ├── scorer
│       │   ├── utils
│       │   ├── geometry
│       │   └── config
│       ├── optimizer
│       │   ├── geometry
│       │   └── config
│       ├── geometry
│       │   └── config
│       └── config
├── utils
│   └── config
├── trajectory (screw-trajectory sampling and bone-density features; NumPy/SimpleITK, PyVista only for mesh membership)
└── __init__ (lazy public exports)
```

Direct third-party dependencies are NumPy, SimpleITK, PyVista, and VTK. Internal imports use package-relative paths.

## FEA Module Dependencies

```text
pedicle_planning.fea
├── abaqus_mesh_assembly
│   └── abaqus_part_writer
├── adaptive_global_face_triangulation
│   └── adaptive_green_face_bucket_builder
├── adaptive_tet_transition_validator
│   └── adaptive_tet_transition_templates
└── adaptive_pattern_specific_green_templates
```

The FEA modules use the Python standard library and internal relative imports. Abaqus files are written as text; Abaqus itself is not a Python dependency and solver execution is outside the package scope.

## Execution Boundaries

The AI API accepts image and geometry objects supplied by the caller. The test suite generates these inputs synthetically. The FEA writers accept nodes, elements, and paths supplied at runtime; no example patient mesh or solver result is bundled.

The package does not contain image segmentation, a clinical case manager, a GUI, screw CAD generation, or an end-to-end solver orchestration layer.
