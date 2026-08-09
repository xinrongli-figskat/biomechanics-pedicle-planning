"""Pure synthetic shared-face triangulation tests."""

import unittest

from pedicle_planning.fea.adaptive_global_face_triangulation import (
    build_required_shared_face_triangulation,
    shared_face_key_from_bbox,
)


class SyntheticFaceTriangulationTests(unittest.TestCase):
    def test_shared_face_key(self):
        bbox = (0, 0, 0, 2, 2, 2)
        self.assertEqual(shared_face_key_from_bbox(bbox, "+x"), ("x", 2, 0, 2, 0, 2))

    def test_two_by_two_face_has_eight_triangles(self):
        triangulation = build_required_shared_face_triangulation(("x", 2, 0, 2, 0, 2))
        self.assertEqual(triangulation.validation_status, "pass")
        self.assertEqual(len(triangulation.subcell_partition), 4)
        self.assertEqual(len(triangulation.triangle_signatures), 8)
