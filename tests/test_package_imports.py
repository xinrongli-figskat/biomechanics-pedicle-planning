"""Package-level import smoke tests."""

import unittest


class PackageImportTests(unittest.TestCase):
    def test_core_packages_import_without_original_project(self):
        import pedicle_planning
        import pedicle_planning.ai
        import pedicle_planning.fea

        self.assertTrue(pedicle_planning.__doc__)

    def test_fea_internal_imports_resolve(self):
        from pedicle_planning.fea import abaqus_mesh_assembly
        from pedicle_planning.fea import adaptive_global_face_triangulation
        from pedicle_planning.fea import adaptive_tet_transition_validator

        self.assertTrue(callable(abaqus_mesh_assembly.build_abaqus_mesh_assembly))
        self.assertTrue(callable(adaptive_global_face_triangulation.shared_face_key_from_bbox))
        self.assertTrue(callable(adaptive_tet_transition_validator.validate_template))
